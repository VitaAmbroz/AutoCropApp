/**
 * Bachelor thesis: Algorithms for automatic image cropping
 * VUT FIT 2018
 * Author: Vít Ambrož (xambro15@stud.fit.vutbr.cz)
 * Supervisor: Doc. Ing. Martin Čadík, Ph. D.
 * File: AutocropSuh.cpp
 * Github repository: https://github.com/VitaAmbroz/AutoCropApp
 */

#include "AutocropSuh.h"

/**
 * Default constructor
 * @param sm Saliency map of original image
 */
AutocropSuh::AutocropSuh(cv::Mat sm) {
	this->x = 0;
	this->y = 0;
	this->width = sm.cols;
	this->height = sm.rows;
	this->salmap = sm;
	// compute saliency value of original image ROI
	this->totalSaliency = this->computeSaliency(0, 0, this->width, this->height); // sum(this->salmap)[0];
}


/**
 * Method for finding ROI with defined width and height
 * @param w Width of ROI
 * @param h Height of ROI
 * @param hStep Horizontal step - number of pixels in x-axis
 * @param vStep Vertical step - number of pixels in y-axis
 */
void AutocropSuh::bruteForceWH(int w, int h, int hStep, int vStep) {
	// input conditions
	if (this->salmap.cols < w || this->salmap.rows < h || hStep < 1 || vStep < 1)
		return;
	
	// save width and height of ROI
	this->width = w;
	this->height = h;

	// values for end of loop
	int endColumn = this->salmap.cols - w;
	int endRow = this->salmap.rows - h;

	// tmp variable for finding best ratio
	double bestRatio = 0.0f;

#pragma omp parallel for
	for (int xx = 0; xx < endColumn; xx += hStep) {
		for (int yy = 0; yy < endRow; yy += vStep) {
			uint32_t salValue = this->computeSaliency(xx, yy, w, h);
			double ratio = (double)salValue / (double)this->totalSaliency;
			// select the best ratio as final crop ROI
			if (ratio > bestRatio) {
				bestRatio = ratio;
				this->x = xx;
				this->y = yy;
			}
		}
	}
}


/**
 * Method for finding ROI with defined scale factor
 * @param scale Scale factor = ratio how the crop should be scaled down against original image
 * @param hStep Horizontal step - number of pixels in x-axis
 * @param vStep Vertical step - number of pixels in y-axis
 */
void AutocropSuh::bruteForceScale(float scale, int hStep, int vStep) {
	// input conditions
	if (scale <= 0.0f || scale >= 1.0f || hStep < 1 || vStep < 1)
		return;

	// save width and height of ROI
	this->width = (int)(scale * this->salmap.cols);
	this->height = (int)(scale * this->salmap.rows);

	// values for end of loop
	int endColumn = this->salmap.cols - this->width;
	int endRow = this->salmap.rows - this->height;

	// tmp variable for finding best ratio
	double bestRatio = 0.0f;

#pragma omp parallel for
	for (int xx = 0; xx < endColumn; xx += hStep) {
		for (int yy = 0; yy < endRow; yy += vStep) {
			uint32_t salValue = this->computeSaliency(xx, yy, this->width, this->height);
			double ratio = (double)salValue / (double)this->totalSaliency;
			// select best ratio as final crop ROI
			if (ratio > bestRatio) {
				bestRatio = ratio;
				this->x = xx;
				this->y = yy;
			}
		}
	}
}


/**
 * Method for finding ROI with defined aspect ratio(width:height)
 * @param w Horizontal parameter for computing width:height ratio
 * @param h Vertical parameter for computing width:height ratio
 * @param treshold 
 * @param hStep Horizontal step - number of pixels in x-axis
 * @param vStep Vertical step - number of pixels in y-axis
 */
void AutocropSuh::bruteForceWHratio(int w, int h, float treshold, int hStep, int vStep) {
	// parameter conditions
	if (w < 1 || h < 1 || treshold <= 0.0f || treshold >= 1.0f || hStep < 1 || vStep < 1)
		return;

	// constant for incrementing width or height until ROI with satisfied treshold is found
	const int SIZE_INCREMENT = (w > h) ? (int)(0.1f * this->salmap.cols) : (int)(0.1f * this->salmap.rows);

	// init temporary variables for width and height
	int tmpW, tmpH;
	if (w > h) {
		tmpW = (int)(MAX_SCALE_SUH * this->salmap.cols);
		tmpH = (int)(tmpW * h / w);
	}
	else {
		tmpH = (int)(MAX_SCALE_SUH * this->salmap.rows);
		tmpW = (int)(tmpH * w / h);
	}

	// init temporary variable for bestRatio value
	float bestRatio = 0.0f;

	// loop until reaching size limits
	while ((tmpW < this->salmap.cols) && (tmpH < this->salmap.rows)) {
		// values for end of loop
		int endColumn = this->salmap.cols - tmpW;
		int endRow = this->salmap.rows - tmpH;
		
		for (int xx = 0; xx < endColumn; xx += hStep) {
			for (int yy = 0; yy < endRow; yy += vStep) {
				uint32_t salValue = this->computeSaliency(xx, yy, tmpW, tmpH);
				double ratio = (double)salValue / (double)this->totalSaliency;

				// check if ROI with defined treshold value was found
				if (ratio > treshold && ratio > bestRatio) {
					bestRatio = ratio;
					this->x = xx;
					this->y = yy;
					this->width = tmpW;
					this->height = tmpH;
				}
			}
		}

		// result ROI was found
		if (bestRatio > 0.0f)
			return;

		// increment size of ROI
		if (w > h) {
			tmpW += SIZE_INCREMENT;
			tmpH = (int)(tmpW * h / w);
		}
		else {
			tmpH += SIZE_INCREMENT;
			tmpW = (int)(tmpH * w / h);
		}
	}

	std::cerr << "ROI with ratio(width:height) = " << w << ":" << h << " was not found. Try to reduce treshold." << std::endl;
}


/**
 * General method that is optimized with greedy algorithm
 * @param treshold Treshold value of minimum saliency energy in final crop ROI
 */
void AutocropSuh::greedyGeneral(float treshold) {
	// parameter conditions
	if (treshold <= 0.0f || treshold >= 1.0f)
		return;

	// init treshold sum
	int tresholdSum = (int)(treshold * this->totalSaliency);

	// constant for recrangle size around peak points
	int RECTANGLE_SIZE = (int)(this->salmap.cols / 20); 
	if (RECTANGLE_SIZE < 1) RECTANGLE_SIZE = 1;

	// init center point
	int centerX = (int)(this->salmap.cols / 2);
	int centerY = (int)(this->salmap.rows / 2);

	// init Rc
	int tmpX1 = centerX - RECTANGLE_SIZE;
	int tmpY1 = centerY - RECTANGLE_SIZE;
	int tmpX2 = centerX + RECTANGLE_SIZE;
	int tmpY2 = centerY + RECTANGLE_SIZE;

	// controls of limits
	if (tmpX1 < 0) tmpX1 = 0;
	if (tmpY1 < 0) tmpY1 = 0;
	if (tmpX2 >= this->salmap.cols) tmpX2 = this->salmap.cols - 1;
	if (tmpY2 >= this->salmap.rows) tmpY2 = this->salmap.rows - 1;
	
	// init temporary saliency sum value
	int currentSaliencySum = this->computeSaliency(tmpX1, tmpY1, (tmpX2 - tmpX1), (tmpY2 - tmpY1));
	// deep copy of original saliency map
	cv::Mat editedSalmap = this->salmap.clone();

	// loop until reaching threshold sum
	while (currentSaliencySum < tresholdSum) {
		// write minimum values to actual Rc
		for (int i = tmpX1; i <= tmpX2; i++) {
			for (int j = tmpY1; j <= tmpY2; j++) {
				// write minimum value <=> 0(black)
				editedSalmap.data[j * this->salmap.cols + i] = 0;
			}
		}
		
		// find maximum saliency point
		double min, max;
		cv::Point min_loc, max_loc;
		cv::minMaxLoc(editedSalmap, &min, &max, &min_loc, &max_loc);

		// check if maximum saliency point is horizontally left or right from actual RC
		if (tmpX1 > max_loc.x) 
			tmpX1 = max_loc.x;
		else if (tmpX2 < max_loc.x)
			tmpX2 = max_loc.x;
		
		// check if maximum saliency point is vertically up or down from actual RC
		if (tmpY1 > max_loc.y) 
			tmpY1 = max_loc.y;
		else if (tmpY2 < max_loc.y) 
			tmpY2 = max_loc.y;

		// update current saliency sum
		currentSaliencySum = this->computeSaliency(tmpX1, tmpY1, (tmpX2 - tmpX1), (tmpY2 - tmpY1));
	}


	// save result ROI parameters
	this->x = tmpX1;
	this->y = tmpY1;
	this->width = tmpX2 - tmpX1;
	this->height = tmpY2 - tmpY1;
}



/**
 * General method for finding ROI with defined threshold(using random generator for width and height of ROI)
 * @param treshold Value of treshold for minimum saliency energy
 */
void AutocropSuh::bruteForceGeneral(float treshold) {
	// input conditions
	if (treshold <= 0.f || treshold >= 1.0f)
		return;

	// construct grid as 1/100 of width and height
	int hStep, vStep;
	hStep = (int)(this->salmap.cols / 100);
	vStep = (int)(this->salmap.rows / 100);
	if (hStep < 1) hStep = 1;
	if (vStep < 1) vStep = 1;

	// apply max scale constant to get minimal width and height of ROI
	int minWidth = (int)(MAX_SCALE_SUH * this->salmap.cols);
	int minHeight = (int)(MAX_SCALE_SUH * this->salmap.rows);

	// values for end of loop
	int endColumn = this->salmap.cols - minWidth;
	int endRow = this->salmap.rows - minHeight;

	// vector for saving candidates for cropping
	std::vector<std::array<int, 4>> candidates;

	// temporary variables
	int tmpWidth, tmpHeight;

	// initialize random number generator
	srand((unsigned int)time(NULL));

	// number of ROIs that will be generated at each position in grid
	const int ROI_NUMBER = 3;

	// loop through grid
	for (int xx = 0; xx < endColumn; xx += hStep) {
		for (int yy = 0; yy < endRow; yy += vStep) {
			for (int j = 0; j < ROI_NUMBER; j++) { // generate 3 ROIs at each location in grid
				tmpWidth = rand() % (this->salmap.cols - xx);
				tmpWidth = (tmpWidth >= minWidth) ? tmpWidth : minWidth;
				tmpHeight = rand() % (this->salmap.rows - yy);
				tmpHeight = (tmpHeight >= minHeight) ? tmpHeight : minHeight;

				uint32_t salValue = this->computeSaliency(xx, yy, tmpWidth, tmpHeight);
				float ratio = (float)((double)salValue / this->totalSaliency);

				// save all ROIs with satisfying treshold
				if (ratio >= treshold) {
					std::array<int, 4> roi = { xx, yy, tmpWidth, tmpHeight };
					candidates.push_back(roi);
				}
			}
		}
	}


	int bestIndex = 0;
	int smallestArea = INT_MAX;
	// find candidate ROI with smallest area
	for (int i = 0; i < candidates.size(); i++) {
		int Sarea = std::get<2>(candidates.at(i)) * std::get<3>(candidates.at(i));
		if (Sarea < smallestArea) {
			smallestArea = Sarea;
			bestIndex = i;
		}
	}

	// save ROI coordinates
	if (candidates.size() > 0) {
		this->x = std::get<0>(candidates.at(bestIndex));
		this->y = std::get<1>(candidates.at(bestIndex));
		this->width = std::get<2>(candidates.at(bestIndex));
		this->height = std::get<3>(candidates.at(bestIndex));
	}
}


/**
 * Method for computing saliency energy of defined ROI
 * @param x1 Horizontal coordinate(x-axis) of ROI top left corner
 * @param y1 Vertical coordinate(y-axis) of ROI top left corner
 * @param w Width of ROI
 * @param h Height of ROI
 * @return Saliency energy of defined ROI
 */
uint32_t AutocropSuh::computeSaliency(int x1, int y1, int w, int h) {
	// position of bottom right corner
	int x2 = x1 + w;
	int y2 = y1 + h;
	
	// input conditions -> return 0 ratio
	if (this->salmap.cols < x2)
		return 0;
	if (this->salmap.rows < y2)
		return 0;

	uint32_t saliency = 0;
	for (int i = x1; i < x2; i++) {
		for (int j = y1; j < y2; j++) {
			// increment score with pixel value [0,255]
			saliency += this->salmap.data[j * this->salmap.cols + i];
		}
	}

	return saliency;
}


/*
* Getter function for x1 position to crop (upper left corner)
*/
int AutocropSuh::getX() {
	return this->x;
}

/*
* Getter function for y1 position to crop (upper left corner)
*/
int AutocropSuh::getY() {
	return this->y;
}

/*
* Getter function for width of cropped area
*/
int AutocropSuh::getWidth() {
	return this->width;
}

/*
* Getter function for height of cropped area
*/
int AutocropSuh::getHeight() {
	return this->height;
}