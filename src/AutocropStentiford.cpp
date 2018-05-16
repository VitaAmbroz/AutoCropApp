/**
 * Bachelor thesis: Algorithms for automatic image cropping
 * VUT FIT 2018
 * Author: Vít Ambrož (xambro15@stud.fit.vutbr.cz)
 * Supervisor: Doc. Ing. Martin Čadík, Ph. D.
 * File: AutocropStentiford.cpp
 * Github repository: https://github.com/VitaAmbroz/AutoCropApp
 */

#include "AutocropStentiford.h"

/**
 * Default constructor
 * @param img Saliency map of original image
 */
AutocropStentiford::AutocropStentiford(cv::Mat sm) {
	this->x = 0;
	this->y = 0;
	this->width = sm.cols;
	this->height = sm.rows;
	this->salMap = sm;
	this->bestScore = 0.0f;
}

/**
* Method for finding optimal cropping ROI with selected Width and Height
* @param hStep Size of horizontal step
* @param vStep Size of vertical step
* @param w Width of result ROI
* @param h Height of result ROI
*/
void AutocropStentiford::brutalForceWH(int w, int h, int hStep, int vStep) {
	// parameters conditions
	if (hStep < 1 || vStep < 1 || w <= 0 || h <= 0 || w > this->salMap.cols || h > this->salMap.rows)
		return;

	// end of loop values
	int endColumn = this->salMap.cols - w;
	int endRow = this->salMap.rows - h;

	// Get ROI with the best average attention score
#pragma omp parallel for
	for (int xx = 0; xx < endColumn; xx += hStep) {
		for (int yy = 0; yy < endRow; yy += vStep) {
			this->computeMaxScore(xx, yy, w, h);
		}
	}
}


/**
* Method for finding optimal cropping ROI with selected zoomFactor
* @param hStep Size of horizontal step
* @param vStep Size of vertical step
* @param zFactor Ratio of size original/result
* (if parameter zFactor is 1.5 => original image is 1.5x bigger than result ROI)
*/
void AutocropStentiford::brutalForceZoomFactor(float zFactor, int hStep, int vStep) {
	// parameters conditions
	if (hStep <= 0 || vStep <= 0 || zFactor <= 1.0)
		return;
	
	// if parameter zFactor is 1.5(original image is 1.5x bigger) => scale is 0.666(2/3 of original image size)
	float scale = 1.0f / zFactor;

	int tmpWidth = (int)(this->salMap.cols * scale);
	int tmpHeight = (int)(this->salMap.rows * scale);

	// values for end of loop
	int endColumn = this->salMap.cols - tmpWidth;
	int endRow = this->salMap.rows - tmpHeight;


	// Get ROI with the best average attention score
#pragma omp parallel for
	for (int xx = 0; xx < endColumn; xx += hStep) {
		for (int yy = 0; yy < endRow; yy += vStep) {
			this->computeMaxScore(xx, yy, tmpWidth, tmpHeight);
		}
	}
}


/**
 * Method for finding optimal cropping ROI in zooming interval(keep aspect ratio)
 * @param from Bottom bound of zoom factor
 * @param to Top bound of zoom factor
 * @param step Step in increasing zoom factor
 * @param hStep Size of horizontal step
 * @param vStep Size of vertical step
 */
void AutocropStentiford::zoomFactorWalk(float from, float to, float step, int hStep, int vStep) {
	// parameters conditions
	if (from >= to || from < 1.0f || step <= 0.0f)
		return;

	// start with zoom factor defined in parameter from
	float zf = from;
	while (zf <= to) {
		// apply method for computing ROI with defined zoom factor
		this->brutalForceZoomFactor(zf, hStep, vStep);
		// increment zoom factor with defined step
		zf += step;
	}
}


/**
 * Method for finding optimal cropping ROI with defined aspect ratio(width:height)
 * @param w Horizontal parameter for computing width:height ratio
 * @param h Vertical parameter for computing width:height ratio
 * @param maxZoomFactor Minimal limit of scale(prevent cropping very small ROI)
 */
void AutocropStentiford::randomWHratio(int w, int h, float maxZoomFactor) {
	// parameter conditions
	if (w < 1 || h < 1 || maxZoomFactor < 1.0f)
		return;
	
	// if parameter zFactor is 1.5(original image is 1.5x bigger) => scale is 0.666(2/3 of original image size)
	float scale = 1.0f / maxZoomFactor;

	// define minimum values
	int minWidth, minHeight;
	if (w > h) {
		minWidth = (int)(scale * this->salMap.cols); // apply scale limit
		minHeight = (int)(minWidth * h / w);
	}
	else {
		minHeight = (int)(scale * this->salMap.rows); // apply scale limit
		minWidth = (int)(minHeight * w / h);
	}

	// temporary variables
	int tmpX, tmpY, tmpWidth, tmpHeight;

	srand((unsigned int)time(NULL));
	// generating random coordinates of top left corner x1,y1 and width(height will be computed)
#pragma omp parallel for
	for (int i = 0; i < ITERATIONS; i++) {
		tmpX = rand() % (this->salMap.cols - minWidth);
		tmpY = rand() % (this->salMap.rows - minHeight);
		tmpWidth = rand() % (this->salMap.cols - tmpX);
		tmpWidth = (tmpWidth >= minWidth) ? tmpWidth : minWidth;
		tmpHeight = (int)(tmpWidth * h / w);

		// compute average attention score for actual ROI
		this->computeMaxScore(tmpX, tmpY, tmpWidth, tmpHeight);
	}
}


/**
* Method for finding optimal cropping using random ROI generator (keeping aspect ratio)
* @param maxZoomFactor Max limit of zoom factor(prevent very small results)
*/
void AutocropStentiford::randomZFWalk(float maxZoomFactor) {
	// parameter condition
	if (maxZoomFactor < 1.0f)
		return;
	
	// temporary variables
	int tmpX, tmpY, tmpWidth, tmpHeight;

	// if parameter zFactor is 1.5(original image is 1.5x bigger) => scale is 0.666(2/3 of original image size)
	float scale = 1.0f / maxZoomFactor;

	// define minimum values
	int minWidth = (int)(scale * this->salMap.cols); 
	int minHeight = (int)(scale * this->salMap.rows); 

	// ratio width/height of original image
	double aspectRatioHW = (double)this->salMap.rows / (double)this->salMap.cols;
	srand((unsigned int)time(NULL));

	// generating random coordinates of top left corner (x1,y1) and width 
#pragma omp parallel for
	for (int i = 0; i < ITERATIONS; i++) {
		tmpX = rand() % (this->salMap.cols - minWidth);
		tmpY = rand() % (this->salMap.rows - minHeight);
		tmpWidth = rand() % (this->salMap.cols - tmpX);
		tmpWidth = (tmpWidth >= minWidth) ? tmpWidth : minHeight;
		tmpHeight = (int)(tmpWidth * aspectRatioHW);

		// compute average attention score for actual ROI
		this->computeMaxScore(tmpX, tmpY, tmpWidth, tmpHeight);
	}
}


/**
* Method for finding optimal cropping using random ROI generator
* @param minWidth The limit of width, there will not be generated lower values than this one
* @param minHeight The limit of height, there will not be generated lower values than this one
*/
void AutocropStentiford::randomWalk(int minWidth, int minHeight) {
	// parameter conditions
	if (minWidth <= 0 || minHeight <= 0 || minWidth > this->salMap.cols || minHeight > this->salMap.rows)
		return;
	
	// temporary variables
	int tmpX, tmpY, tmpWidth, tmpHeight;
	srand((unsigned int)time(NULL));

	// generating random coordinates of top left corner (x1,y1) and also width and height
#pragma omp parallel for
	for (int i = 0; i < ITERATIONS; i++) {
		tmpX = rand() % (int)(this->salMap.cols - minWidth);
		tmpY = rand() % (int)(this->salMap.rows - minHeight);
		tmpWidth = rand() % (this->salMap.cols - tmpX);
		tmpHeight = rand() % (this->salMap.rows - tmpY);
		// min limit for width and height
		tmpWidth = (tmpWidth > minWidth) ? tmpWidth : minWidth;
		tmpHeight = (tmpHeight > minHeight) ? tmpHeight : minHeight;

		// apply method for getting roi with random parametres
		this->computeMaxScore(tmpX, tmpY, tmpWidth, tmpHeight);
	}
}


/**
* Method for computing maximal average score in defined ROI
* @param hStep size of horizontal step
* @param vStep size of vertical step
* @param w Width of result ROI
* @param h Height of result ROI
*/
void AutocropStentiford::computeMaxScore(int x1, int y1, int w, int h) {
	double actualScore = 0.0f;

	int x2 = x1 + w;
	int y2 = y1 + h;

	// input condition
	if (this->salMap.cols < x2)
		return;
	if (this->salMap.rows < y2)
		return;

	//uint8_t valPixel;
	for (int i = x1; i < x2; i++) {
		for (int j = y1; j < y2; j++) {
			//valPixel = this->salMap.data[j * salMap.cols + i];
			//valPixel = this->salMap.at<uint8_t>(j, i); // slower
			
			// increment score with pixel value [0,255]
			actualScore += this->salMap.data[j * this->salMap.cols + i];;
		}
	}

	// compute average attention score for 1px
	actualScore /= (w * h);
	// if actual attention score is better, save this ROI
	if (actualScore > this->bestScore) {
		this->bestScore = actualScore;
		this->x = x1;
		this->y = y1;
		this->width = w;
		this->height = h;
	}
}


/*
* Getter function for x1 position to crop
*/
int AutocropStentiford::getX() {
	return this->x;
}

/*
* Getter function for y1 position to crop
*/
int AutocropStentiford::getY() {
	return this->y;
}

/*
* Getter function for width of cropped area
*/
int AutocropStentiford::getWidth() {
	return this->width;
}

/*
* Getter function for height of cropped area
*/
int AutocropStentiford::getHeight() {
	return this->height;
}