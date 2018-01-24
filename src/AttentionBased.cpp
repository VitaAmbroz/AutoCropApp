#include "AttentionBased.h"

using namespace std;
using namespace cv;


AttentionBased::AttentionBased(cv::Mat img) {
	this->x = 0;
	this->y = 0;
	this->width = img.cols;
	this->height = img.rows;
	this->image = img;
	this->bestScore = 0.0f;
}


/**
* Method for finding optimal cropping ROI with selected Width and Height
* @param hStep Size of horizontal step
* @param vStep Size of vertical step
* @param w Width of result ROI
* @param h Height of result ROI
*/
void AttentionBased::brutalForceWH(int hStep, int vStep, int w, int h) {
	// parameters conditions
	if (hStep <= 0 || vStep <= 0 || w <= 0 || h <= 0 || w > this->image.cols || h > this->image.rows)
		return;

	// end of loop values
	int endColumn = this->image.cols - w;
	int endRow = this->image.rows - h;

	// Get ROI with the best attention score
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
void AttentionBased::brutalForceZoomFactor(int hStep, int vStep, float zFactor) {
	// parameters conditions
	if (hStep <= 0 || vStep <= 0 || zFactor <= 1.0)
		return;
	
	// reverse zoom factor (ratio of size result/original)
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
 * Method for finding optimal cropping roi in interval of zooming(same aspect ratio)
 * @param hStep Size of horizontal step
 * @param vStep Size of vertical step
 * @param from Bottom bound of zoom factor
 * @param to Top bound of zoom factor
 * @param step Step in increasing zoom factor
 */
void AttentionBased::zoomFactorWalk(int hStep, int vStep, float from, float to, float step) {
	// parameters conditions
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
* Method for finding optimal cropping using random ROI generator (keeping aspect ratio)
* @param iterations Number of generated ROIs
* @param maxZoomFactor Max limit of zoom factor(prevent very small results)
*/
void AttentionBased::randomZFWalk(int iterations, float maxZoomFactor) {
	// parameter condition
	if (maxZoomFactor < 1.0f)
		return;
	
	// temporary variables
	int tmpX, tmpY, tmpWidth, tmpHeight;

	// reverse zoom factor
	// if parameter zFactor is 1.5(original image is 1.5x bigger) => reverseZF is 0.666(2/3 of original image size)
	float reverseZFactor = 1.0f / maxZoomFactor;

	// ratio width/height of original image
	double aspectRatioHW = (double)this->image.rows / (double)this->image.cols;
	srand((unsigned int)time(NULL));

	// generating random coordinates of top left corner (x1,y1)
	for (int i = 0; i < iterations; i++) {
		tmpX = rand() % (int)(this->image.cols * (1.0f - reverseZFactor));
		tmpY = rand() % (int)(this->image.rows * (1.0f - reverseZFactor));
		tmpWidth = rand() % (this->image.cols - tmpX);
		tmpWidth = (int)((tmpWidth > (this->image.cols * reverseZFactor)) ? tmpWidth : (this->image.cols * reverseZFactor));
		tmpHeight = (int)(tmpWidth * aspectRatioHW);

		// apply method for getting roi with random parametres
		this->computeMaxScore(tmpX, tmpY, tmpWidth, tmpHeight);
	}
}


/**
* Method for finding optimal cropping using random ROI generator
* @param iterations Number of generated ROIs
* @param minWidth The limit of width, there will not be generated lower values than this one
* @param minHeight The limit of height, there will not be generated lower values than this one
*/
void AttentionBased::randomWalk(int iterations, int minWidth, int minHeight) {
	// parameter condition
	if (minWidth <= 0 || minHeight <= 0 || minWidth > this->image.cols || minHeight > this->image.rows)
		return;
	
	// temporary variables
	int tmpX, tmpY, tmpWidth, tmpHeight;
	srand((unsigned int)time(NULL));

	// generating random coordinates of top left corner (x1,y1) and width+height
	for (int i = 0; i < iterations; i++) {
		tmpX = rand() % (int)(this->image.cols - minWidth);
		tmpY = rand() % (int)(this->image.rows - minHeight);
		tmpWidth = rand() % (this->image.cols - tmpX);
		tmpHeight = rand() % (this->image.rows - tmpY);
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
void AttentionBased::computeMaxScore(int x1, int y1, int w, int h) {
	double actualScore = 0.0f;

	int x2 = x1 + w;
	int y2 = y1 + h;

	// input condition
	if (this->image.cols < x2)
		return;
	if (this->image.rows < y2)
		return;

	//uint8_t valPixel;
	for (int i = x1; i < x2; i++) {
		for (int j = y1; j < y2; j++) {
			// get pixel value [0,255]
			//valPixel = this->image.data[j * image.cols + i];
			//valPixel = this->image.at<uint8_t>(j, i); // slower
			
			// increment score with pixel value [0,255]
			actualScore += this->image.data[j * image.cols + i];;
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