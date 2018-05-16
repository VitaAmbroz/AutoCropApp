/**
 * Bachelor thesis: Algorithms for automatic image cropping
 * VUT FIT 2018
 * Author: Vít Ambrož (xambro15@stud.fit.vutbr.cz)
 * Supervisor: Doc. Ing. Martin Čadík, Ph. D.
 * File: AutocropFang.cpp
 * Github repository: https://github.com/VitaAmbroz/AutoCropApp
 */

#include "AutocropFang.h"

/**
* Constructor for cropping algorithm(Fang et al. 2014)
* @param img Original image to be cropped
* @param sMap Saliency map of original image(Margolin et al. 2013)
* @param trainedModelPath Filepath for trained Visual Composition model
*/
AutocropFang::AutocropFang(cv::Mat img, cv::Mat sMap, std::string trainedModelPath) {
	this->x = 0;
	this->y = 0;
	this->width = img.cols;
	this->height = img.rows;

	this->image = img;	// original image
	this->salMap = sMap;	// saliency map of original image
	// normalize saliency map (cv_32F) to matrix CV_8UC1 because of faster access to pixel values
	cv::normalize(sMap, this->salMapNorm, 0, 255, NORM_MINMAX, CV_8UC1);
	// compute total saliency energy of input image
	this->totalSaliencyEnergy = this->computeSaliencyEnergy(0, 0, img.cols, img.rows);
	this->gradient = this->getGradient(img);	// generate image gradient
	this->compos.loadTrainedModel(trainedModelPath);
}


/**
* Method for finding optimal cropping window with defined width and height, walking through grid made from hStep and vStep
* @param w Width of cropping window
* @param h Height of cropping window
* @param hStep Horizontal step - number of pixels in x-axis
* @param vStep Vertical step - number of pixels in y-axis
*/
void AutocropFang::WHCrop(int w, int h, int hStep, int vStep) {
	// parameter conditions
	if ( w <= 0 || h <= 0 || w >= this->image.cols || h >= this->image.rows || hStep < 1 || vStep < 1)
		return;
	
	// init treshold - find ROI with large content first
	float treshold = SALIENCY_TRESHOLD_INIT;
	// vector for saving generated candidates for cropping
	std::vector<std::array<int, 4>> candidates;

	// save width and height of cropping window
	this->width = w;
	this->height = h;

	// values for end of loop
	int endColumn = this->image.cols - w;
	int endRow = this->image.rows - h;

	while (candidates.size() < CANDIDATES_COUNT) {  // loop until there will be enough candidate ROIs
		for (int x1 = 0; x1 < endColumn; x1 += hStep) {
			for (int y1 = 0; y1 < endRow; y1 += vStep) {
				// check if candidate ROI has satisfactory saliency energy
				if (this->candidateContentPreserv(x1, y1, w, h, treshold)) {
					std::array<int, 4> roi = { x1, y1, w, h };
					candidates.push_back(roi);
				}
			}
		}
		treshold -= 0.1f; // decrease treshold if there was not enough candidate ROIs
	}

	// find best ROI according to boundary simplicity cue and save values(x, y, width, height)
	this->getBestCandidate(candidates);
}


/**
* Method for finding optimal cropping ROI(keeping aspect ratio), walking through grid made from hStep and vStep
* @param scale Zoom factor for cropping window(prevent very small results)
* (if parameter scale is 0.5 => ROI will be 0.5x smaller than original)
* @param hStep Horizontal step - number of pixels in x-axis
* @param vStep Vertical step - number of pixels in y-axis
*/
void AutocropFang::scaleCrop(float scale, int hStep, int vStep) {
	// parameter conditions
	if (scale < 0.0f || scale >= 1.0f || hStep < 1 || vStep < 1)
		return;

	// init treshold - find ROI with large content first
	float treshold = SALIENCY_TRESHOLD_INIT;
	// vector for saving generated candidates for cropping
	std::vector<std::array<int, 4>> candidates;

	// apply scale to width and height
	int w = (int)(this->image.cols * scale);
	int h = (int)(this->image.rows * scale);
	this->width = w;
	this->height = h;

	// values for end of loop
	int endColumn = this->image.cols - w;
	int endRow = this->image.rows - h;

	while (candidates.size() < CANDIDATES_COUNT) {  // loop until there will be enough candidate ROIs
		for (int x1 = 0; x1 < endColumn; x1 += hStep) {
			for (int y1 = 0; y1 < endRow; y1 += vStep) {
				// check if candidate ROI has satisfactory saliency energy
				if (this->candidateContentPreserv(x1, y1, w, h, treshold)) {
					std::array<int, 4> xy = { x1, y1, w, h };
					candidates.push_back(xy);
				}
			}
		}
		treshold -= 0.1f; // decrease treshold if there was not enough candidate ROIs
	}

	// find best ROI according to boundary simplicity cue and save values(x, y, width, height)
	this->getBestCandidate(candidates);
}


/**
* Method for finding optimal cropping with defined Width:Height ratio
* @param w Horizontal parameter for computing width:height ratio
* @param h Vertical parameter for computing width:height ratio
* @param hStep Horizontal step - number of pixels in x-axis
* @param vStep Vertical step - number of pixels in y-axis
*/
void AutocropFang::WHratioCrop(int w, int h, int hStep, int vStep) {
	// parameter conditions
	if (w < 1 || h < 1 || hStep < 1 || vStep < 1)
		return;

	// vector for saving generated candidates for cropping
	std::vector<std::array<int, 4>> candidates;
	// init content preservation(saliency) treshold empirically to 0.66
	const float TRES = 0.66f;
	// constant for incrementing width or height until enough candidates are generated
	const int SIZE_INCREMENT = (w > h) ? (int)(0.05f * this->image.cols) : (int)(0.05f * this->image.rows);

	// init temporary variables for width and height
	int tmpW, tmpH;
	if (w > h) {
		tmpW = (int)(MAX_SCALE_FANG * this->image.cols); // apply constant for minimum size to avoid very small ROIs
		tmpH = (int)(tmpW * h / w);
	}
	else {
		tmpH = (int)(MAX_SCALE_FANG * this->image.rows); // apply constant for minimum size to avoid very small ROIs
		tmpW = (int)(tmpH * w / h);
	}

	// loop until reaching size limits or enough candidates are generated
	while ((tmpW < this->image.cols) && (tmpH < this->image.rows) && (candidates.size() < CANDIDATES_COUNT)) {
		// values for end of loop
		int endColumn = this->image.cols - tmpW;
		int endRow = this->image.rows - tmpH;

		for (int x1 = 0; x1 < endColumn; x1 += hStep) {
			for (int y1 = 0; y1 < endRow; y1 += vStep) {
				// check if candidate ROI has satisfactory saliency energy
				if (this->candidateContentPreserv(x1, y1, tmpW, tmpH, TRES)) {
					std::array<int, 4> xy = { x1, y1, tmpW, tmpH };
					candidates.push_back(xy);
				}
			}
		}

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

	// treshold that was set empirically was too high, no ROIs with this ratio were found
	if (candidates.size() <= 0) {
		std::cerr << "ROI with ratio(width:height) = " << w << ":" << h << " was not found. Treshold for Content preservation model was too high." << std::endl;
		return;
	}

	// find best ROI according to boundary simplicity and visual composition models
	this->getBestCandidate(candidates);
}


/**
* Method for finding cropping ROI according to article.
* ROI is computed for each position in grid using random generator of width and height. 
*/
void AutocropFang::randomGridCrop() {
	// sample grid as 1/100 of image size
	int hGrid = this->image.cols / 100;
	int vGrid = this->image.rows / 100;
	if (hGrid < 1) hGrid = 1;
	if (vGrid < 1) vGrid = 1;

	// init minimum size of ROI
	int minWidth = (int)(this->image.cols * MAX_SCALE_FANG);
	int minHeight = (int)(this->image.rows * MAX_SCALE_FANG);
	// values for end of loop
	int endCol = this->image.cols - minWidth;
	int endRow = this->image.rows - minHeight;
	// init temporary variables for width and height
	int tmpWidth = 0;
	int tmpHeight = 0;

	// vector for saving generated candidates for cropping
	std::vector<std::array<int, 4>> candidates;
	// initialize random number generator
	srand((unsigned int)time(NULL));

	// empirically set treshold for content preservation model to 0.7 (originally it was 0.4, but that very often misses main subjects)
	const float SAL_TRESHOLD = 0.7f;
	// number of ROIs that will be generated at each position in grid
	const int ROI_NUMBER = 10;

	for (int x1 = 0; x1 < endCol; x1 += hGrid) {
		for (int y1 = 0; y1 < endRow; y1 += vGrid) {
			int i = 0;
			while (i < ROI_NUMBER) {	// generate defined number ROIs at each location
				// generate random width and height
				tmpWidth = rand() % (this->image.cols - x1);
				tmpHeight = rand() % (this->image.rows - y1);
				// condition for very small ROIs -> apply minimal size
				if (tmpWidth < minWidth) tmpWidth = minWidth;
				if (tmpHeight < minHeight) tmpHeight = minHeight;

				// check if candidate ROI has satisfactory saliency energy
				if (this->candidateContentPreserv(x1, y1, tmpWidth, tmpHeight, SAL_TRESHOLD)) {
					std::array<int, 4> xy = { x1, y1, tmpWidth, tmpHeight };
					candidates.push_back(xy);
				}
				i++;
			}
		}
	}

	// find best ROI according to boundary simplicity cue and save values(x, y, width, height)
	this->getBestCandidate(candidates);
}


/**
* Method for computing scores from Boundary simplicity and Visual composition models and finds best cropping ROI
* @param candidates Vector of cropping candidate ROIs
*/
void AutocropFang::getBestCandidate(std::vector<std::array<int, 4>> candidates) {
	int bestIndex = 0;	// init index of best candidate
	int bestOrder = INT32_MAX;	//  
	std::vector<float> boundaryScore;
	std::vector<float> composScore;
	std::vector<int> boundaryOrder;
	std::vector<int> composOrder;

	for (int i = 0; i < candidates.size(); i++) {
		/* Boundary Simplicity Part */
		boundaryScore.push_back(this->computeBoundarySimplicity(std::get<0>(candidates.at(i)),
			std::get<1>(candidates.at(i)),
			std::get<2>(candidates.at(i)),
			std::get<3>(candidates.at(i))));

		/* Visual Composition Part */
		composScore.push_back(this->computeVisualComposition(std::get<0>(candidates.at(i)),
			std::get<1>(candidates.at(i)),
			std::get<2>(candidates.at(i)),
			std::get<3>(candidates.at(i))));
	}


	for (int i = 0; i < boundaryScore.size(); i++) {
		int order = 1;
		for (int j = 0; j < boundaryScore.size(); j++) 
			if (boundaryScore.at(i) > boundaryScore.at(j)) // lower value means better result in BS model
				order++;

		// weight of normalized boundary simplicity model
		order *= WEIGHT_BOUNDARY;
		boundaryOrder.push_back(order);
	}

	for (int i = 0; i < composScore.size(); i++) {
		int order = 1;
		for (int j = 0; j < composScore.size(); j++)
			if (composScore.at(i) > composScore.at(j)) // lower value means better result in VC model
				order++;

		// weight of normalized visual composition model
		order *= WEIGHT_COMPOS;
		composOrder.push_back(order);
	}

	for (int i = 0; i < candidates.size(); i++) {
		if (bestOrder > (boundaryOrder.at(i) + composOrder.at(i))) {
			bestOrder = boundaryOrder.at(i) + composOrder.at(i);
			bestIndex = i;
		}
	}

	// save coordinates of top left corner => bestIndex was saved above
	if (candidates.size() > 0) {
		this->x = std::get<0>(candidates.at(bestIndex));
		this->y = std::get<1>(candidates.at(bestIndex));
		this->width = std::get<2>(candidates.at(bestIndex));
		this->height = std::get<3>(candidates.at(bestIndex));
	}
}


/**
* Method for finding cropping ROI according to article.
* @param x1 Horizontal coordinate(x-axis) of top left corner
* @param y1 Vertical coordinate(y-axis) of top left corner
* @param w Width of ROI
* @param h Height of ROI
* @param treshold Ratio that is satysfying for Content preservation model
* @return Returns true if there ratio is higher than treshold, else returns false.
*/
bool AutocropFang::candidateContentPreserv(int x1, int y1, int w, int h, float treshold) {
	if (this->image.cols <= (x1 + w) || this->image.rows <= (y1 + h))
		return false;
	
	// compute saliency energy of actual window
	uint32_t saliencyScore = this->computeSaliencyEnergy(x1, y1, w, h);

	// ratio of saliency energy cropping ROI/original image
	float saliencyRatio = (float)saliencyScore / this->totalSaliencyEnergy;

	// preserv ROIs with satisfactory saliency energy
	if (saliencyRatio > treshold)
		return true;

	return false;
}


/**
 * Method for computing saliency energy of selected ROI
 * @param x1 Horizontal coordinate(x-axis) of ROI top left corner
 * @param y1 Vertical coordinate(y-axis) of ROI top left corner
 * @param w Width of ROI
 * @param h Height of ROI
 * @return Saliency energy of defined ROI
 */
uint32_t AutocropFang::computeSaliencyEnergy(int x1, int y1, int w, int h) {
	uint32_t score = 0;

	// right bottom corner of ROI
	int x2 = x1 + w;
	int y2 = y1 + h;

	// input condition
	if (x2 > this->salMapNorm.cols)
		return 0;
	if (y2 > this->salMapNorm.rows)
		return 0;

	uint8_t valPixel = 0;
	for (int i = x1; i < x2; i++) {
		for (int j = y1; j < y2; j++) {
			// get pixel value [0,255]
			valPixel = this->salMapNorm.data[j * salMap.cols + i];
			// get pixel value <0, 1>
			//valPixel = this->salMap.at<float>(j, i);

			// increment score with pixel value
			score += valPixel;
		}
	}

	return score;
}


/* src : https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html */
/**
* Method for generating gradient of image -> big values for edges of objects
* @param img Original image
* @return Gradient of original image with big values for edges.
*/
cv::Mat AutocropFang::getGradient(cv::Mat img) {
	Mat src = img.clone();	// copy original image
	Mat src_gray;	// mat for converting original to gray
	Mat grad;	// final gradient image (output)
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	// conditions of blur for large image => better result
	int sigma = 2;
	if (img.cols >= 3000 || img.rows >= 3000)
		sigma = 8;
	else if (img.cols >= 2000 || img.rows >= 2000)
		sigma = 5;
	else if (img.cols >= 1200 || img.rows >= 1200)
		sigma = 3;
	// apply a GaussianBlur to our image to reduce the noise
	cv::GaussianBlur(src, src, Size(0, 0), sigma, sigma, BORDER_DEFAULT);

	// Convert it to gray
	cv::cvtColor(src, src_gray, CV_BGR2GRAY);

	// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	// Calculate the "derivatives" in x and y directions
	cv::Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	cv::Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	// convert our partial results back to CV_8U
	cv::convertScaleAbs(grad_x, abs_grad_x);
	cv::convertScaleAbs(grad_y, abs_grad_y);

	// Total Gradient (approximate)
	cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	// normalize it in range [0,255]
	cv::normalize(grad, grad, 0, 255, NORM_MINMAX, CV_8UC1);

	return grad;
}


/**
 * Method for computing boundary simplicity value of selected ROI
 * @param x1 Horizontal coordinate(x-axis) of ROI top left corner
 * @param y1 Vertical coordinate(y-axis) of ROI top left corner
 * @param w Width of ROI
 * @param h Height of ROI
 * @return Boundary simplicity value of defined ROI
 */
float AutocropFang::computeBoundarySimplicity(int x1, int y1, int w, int h) {
	uint32_t score = 0;
	uint32_t pixels = 0;

	// right bottom corner of ROI
	int x2 = x1 + w - 1;
	int y2 = y1 + h - 1;

	// input condition
	if (this->gradient.cols <= x2)
		return 0;
	if (this->gradient.rows <= y2)
		return 0;


	for (int i = x1; i < x2; i++) {
		// increment score with pixel value [0,255]
		// top bound (2 pixels wide)
		score += (uint8_t)this->gradient.data[y1 * this->gradient.cols + i];
		score += (uint8_t)this->gradient.data[(y1 + 1) * this->gradient.cols + i];
		// bottom bound (2 pixels wide)
		score += (uint8_t)this->gradient.data[(y2 - 1) * this->gradient.cols + i];
		score += (uint8_t)this->gradient.data[y2 * this->gradient.cols + i];
		
		// increment number of pixels
		pixels += 4;
	}
	
	for (int j = (y1 + 2); j < (y2 - 2); j++) {
		// increment score with pixel value [0,255]
		// left bound (2 pixels wide)
		score += (uint8_t)this->gradient.data[j * this->gradient.cols + x1];
		score += (uint8_t)this->gradient.data[j * this->gradient.cols + (x1 + 1)];
		// right bound (2 pixels wide)
		score += (uint8_t)this->gradient.data[j * this->gradient.cols + (x2 - 1)];
		score += (uint8_t)this->gradient.data[j * this->gradient.cols + x2];
		
		// increment number of pixels
		pixels += 4;
	}

	return (float)score / (float)pixels;
}


/**
* This methods computes score of ROI in Visual Composition model
* @param x1 Horizontal coordinate(x-axis) of ROI top left corner
* @param y1 Vertical coordinate(y-axis) of ROI top left corner
* @param w Width of ROI
* @param h Height of ROI
* @return Visual composition score of defined ROI
*/
float AutocropFang::computeVisualComposition(int x1, int y1, int w, int h) {
	/* vyrazne zpomalovalo - nyni nacitat jen jednou na zacatku
	CompositionModel comp;
	comp.loadTrainedModel(this->trainedModelPath);*/
	
	// create rectangle of defined ROI
	cv::Rect rect = cv::Rect(x1, y1, w, h);
	cv::Mat cropCandidate;
	cropCandidate = this->salMap(rect);	// crop this ROI

	// compute feature vector(size of 21 or 85)
	cv::Mat fvec = this->compos.getFeatureVector(cropCandidate);
	// compute score -> the lowest value means best score
	float score = this->compos.classifyComposition(fvec);

	return score;
}


/*
* Getter function for x1 coordinate of cropping ROI (top left corner)
*/
int AutocropFang::getX() {
	return this->x;
}

/*
* Getter function for y1 coordinate of cropping ROI (upper left corner)
*/
int AutocropFang::getY() {
	return this->y;
}

/*
* Getter function for width of cropping ROI
*/
int AutocropFang::getWidth() {
	return this->width;
}

/*
* Getter function for height of cropping ROI
*/
int AutocropFang::getHeight() {
	return this->height;
}
