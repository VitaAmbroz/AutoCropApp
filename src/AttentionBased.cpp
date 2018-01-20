#include "AttentionBased.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


AttentionBased::AttentionBased(cv::Mat img) {
	this->x = 0;
	this->y = 0;
	this->width = 0;
	this->height = 0;
	this->bestScore = 0.0f;
	this->image = img;
	this->pixelPtr = (uint8_t*)img.data;
}


/**
 * Method for finding optimal cropping roi in interval of zooming
 * @param hStep size of horizontal step
 * @param vStep size of vertical step
 * @param from bottom bound of zoom factor
 * @param to top bound of zoom factor
 * @param step step in increasing zoom factor
 */
void AttentionBased::zoomFactorWalk(int hStep, int vStep, float from, float to, float step) {
	// input conditions
	if (from >= to || from < 1.0f || step <= 0.0f)
		return;

	// start with zoom factor defined in parameter from
	float zf = from;
	while (zf <= to) {
		// apply method for getting roi with defined zoom factor
		this->brutalForceZoomFactor(hStep, vStep, zf);
		// increment zoom factor with its step
		zf += step;
	}
}


/**
* Method for finding optimal cropping using random ROI generator
* @param iterations number of generated ROIs
* @param maxZoomFactor max limit of zoom factor(prevent very small results)
*/
void AttentionBased::randomWalk(int iterations, float maxZoomFactor) {
	// temporary variables
	int tmpX, tmpY, tmpWidth, tmpHeight;

	// reverse zoom factor
	// if parameter zFactor is 1.5(original image is 1.5x bigger) => reverseZF is 0.666(2/3 of original image size)
	float reverseZFactor = 1.0f / maxZoomFactor;

	double aspectRatioHW = (double)this->image.rows / (double)this->image.cols;
	srand((unsigned int)time(NULL));
	for (int i = 0; i < iterations; i++) {
		tmpX = rand() % (this->image.cols / 2);
		tmpY = rand() % (this->image.rows / 2);
		tmpWidth = rand() % (this->image.cols - tmpX) + 1;
		tmpWidth = (int)((tmpWidth > (this->image.cols * reverseZFactor)) ? tmpWidth : (this->image.cols * reverseZFactor));
		tmpHeight = (int)(tmpWidth * aspectRatioHW);

		// apply method for getting roi with random parametres
		this->computeMaxScore(tmpX, tmpY, tmpWidth, tmpHeight);
	}
}


/**
* Method for finding optimal cropping ROI with selected zoomFactor
* @param hStep size of horizontal step
* @param vStep size of vertical step
* @param zFactor ratio original/result (if parameter zFactor is 1.5 => original image is 1.5x bigger than result ROI)
*/
void AttentionBased::brutalForceZoomFactor(int hStep, int vStep, float zFactor) {
	// input conditions
	if (hStep <= 0 || vStep <= 0 || zFactor <= 1.0)
		return;
	
	// reverse zoom factor
	// if parameter zFactor is 1.5(original image is 1.5x bigger) => reverseZF is 0.666(2/3 of original image size)
	float reverseZFactor = 1.0f / zFactor;

	int tmpWidth = (int)(this->image.cols * reverseZFactor);
	int tmpHeight = (int)(this->image.rows * reverseZFactor);

	// values for end of loop
	int endColumn = this->image.cols - tmpWidth;
	int endRow = this->image.rows - tmpHeight;

	// Get ROI with the best attention score
	for (int xx = 0; xx < endColumn; xx += hStep) {
		for (int yy = 0; yy < endRow; yy += vStep) {
			this->computeMaxScore(xx, yy, tmpWidth, tmpHeight);
		}
	}
}


/**
* Method for finding optimal cropping ROI with selected Width and Height
* @param hStep size of horizontal step
* @param vStep size of vertical step
* @param width Width of result ROI
* @param width Height of result ROI
*/
void AttentionBased::brutalForceWH(int hStep, int vStep, int width, int height) {
	// input conditions
	if (hStep <= 0 || vStep <= 0 || width <= 0 || height <= 0)
		return;

	// end of loop values
	int endColumn = this->image.cols - width;
	int endRow = this->image.rows - height;

	// Get ROI with the best attention score
	for (int xx = 0; xx < endColumn; xx += hStep) {
		for (int yy = 0; yy < endRow; yy += vStep) {
			this->computeMaxScore(xx, yy, width, height);
		}
	}
}


/**
* Method for computing maximal average score in defined ROI
* @param hStep size of horizontal step
* @param vStep size of vertical step
* @param width Width of result ROI
* @param width Height of result ROI
*/
void AttentionBased::computeMaxScore(int x1, int y1, int width, int height) {
	double actualScore = 0.0f;

	int x2 = x1 + width;
	int y2 = y1 + height;

	// input condition
	if (this->image.cols < x2)
		return;
	if (this->image.rows < y2)
		return;

	uint8_t valPixel;
	for (int i = x1; i < x2; i++) {
		for (int j = y1; j < y2; j++) {
			// get pixel value [0,255]
			valPixel = pixelPtr[j * image.cols + i];
			//valPixel = this->image.at<uint8_t>(j, i);
			
			// increment score with pixel value [0,255]
			actualScore += valPixel;
		}
	}

	// compute average attention score 
	actualScore /= (width * height);
	// if actual attention score is better, save this ROI
	if (actualScore > this->bestScore) {
		this->bestScore = actualScore;
		this->x = x1;
		this->y = y1;
		this->width = width;
		this->height = height;
	}
}


/*
* Getter function for x1 position to crop
*/
int AttentionBased::getX() {
	return this->x;
}

/*
* Getter function for y1 position to crop
*/
int AttentionBased::getY() {
	return this->y;
}

/*
* Getter function for width of cropped area
*/
int AttentionBased::getWidth() {
	return this->width;
}

/*
* Getter function for height of cropped area
*/
int AttentionBased::getHeight() {
	return this->height;
}



/* Random definiton of area for cropping */
//void AttentionBased::randomCrop() {
//	srand(time(NULL));
//
//	this->x = rand() % (this->image.size().width / 2);
//	this->y = rand() % (this->image.size().height / 2);
//	this->width = rand() % (this->image.size().width - this->x) + 1;
//	this->height = rand() % (this->image.size().height - this->y) + 1;
//}

//void AttentionBased::randomBetterCrop() {
//	srand(time(NULL));
//
//	this->x = rand() % (this->image.size().width / 3);
//	this->y = rand() % (this->image.size().height / 3);
//	this->width = this->image.size().width - this->x - (rand() % (this->x));
//	this->height = this->image.size().height - this->y - (rand() % (this->y));
//}