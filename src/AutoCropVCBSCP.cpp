#include "AutoCropVCBSCP.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


AutoCropVCBSCP::AutoCropVCBSCP(cv::Mat img, cv::Mat sMap) {
	this->x = 0;
	this->y = 0;
	this->width = img.cols;
	this->height = img.rows;

	this->image = img;	// original image
	this->salMap = sMap;	// saliency map of original image
	this->gradient = this->getGradient(img);	// generate image gradient
	this->totalSaliencyEnergy = this->computeSaliencyEnergy(0, 0, (sMap.cols - 1), (sMap.rows - 1));
}


/* src : https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html */
/**
 * Method for generating gradient of image -> big values for edges of objects
 * @param img Original image
 * @return Gradient of original image with big values for edges.
 */
cv::Mat AutoCropVCBSCP::getGradient(cv::Mat img)
{
	Mat src = img.clone();	// copy original image
	Mat src_gray;	// mat for converting original to gray
	Mat grad;	// final gradient image (output)
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	// apply a GaussianBlur to our image to reduce the noise (kernel size = 3)
	cv::GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

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

	return grad;
}


/**
* Method for finding optimal cropping window with defined width and height, using random ROI generator
* @param iterations Number of generated ROIs
* @param w Width of cropping window
* @param h Height of cropping window
*/
void AutoCropVCBSCP::randomWH(int iterations, int w, int h) {
	// parameter conditions
	if ( w <= 0 || h <= 0 || w >= this->image.cols || h >= this->image.rows)
		return;
	
	// temporary variables
	int tmpX, tmpY;
	uint32_t saliencyScore = 0;
	float saliencyRatio = 0.0f;

	// vector for saving generated candidates for cropping
	std::vector<std::array<int, 2>> candidates;

	// save width and height of cropping window
	this->width = w;
	this->height = h;

	srand((unsigned int)time(NULL));

	for (int i = 0; i < iterations; i++) {
		// generate 
		tmpX = rand() % (this->image.cols - w);
		tmpY = rand() % (this->image.rows - h);
		// prevent conditions
		if (this->salMap.cols < (tmpX + this->width)) continue;
		if (this->salMap.rows < (tmpY + this->height)) continue;

		// compute saliency energy of actual window
		saliencyScore = this->computeSaliencyEnergy(tmpX, tmpY, this->width, this->height);

		/* Content Preservation Part */
		saliencyRatio = (float)saliencyScore / (float)this->totalSaliencyEnergy;
		if (saliencyRatio > 0.4f) {
			std::array<int, 2> xy = { tmpX, tmpY };
			candidates.push_back(xy);
		}
		/* Content Preservation Part End */
	}

	// init variables for Boundary simplicity cue
	uint32_t tmpScore = DEF_UINT32_MAX;
	uint32_t bestBoundaryScore = DEF_UINT32_MAX;
	int bestIndex = 0;

	for (int i = 0; i < candidates.size(); i++) {
		/* Boundary Simplicity Part */
		tmpScore = this->computeBoundarySimplicity(std::get<0>(candidates.at(i)),
												std::get<1>(candidates.at(i)),
												this->width,
												this->height);
		// finding the smallest score (pixel in boundary of objects has bigger values)
		if (tmpScore < bestBoundaryScore) {
			// save index in candidates and its score
			bestIndex = i;
			bestBoundaryScore = tmpScore;
		}
		/* Boundary Simplicity Part End */
	}

	// save coordinates of top left corner => bestIndex was saved above
	if (candidates.size() > 0) {
		this->x = std::get<0>(candidates.at(bestIndex));
		this->y = std::get<1>(candidates.at(bestIndex));
	}
}



/**
* Method for finding optimal cropping using random ROI generator(keeping aspect ratio)
* @param iterations Number of generated ROIs
* @param maxZoomFactor Zoom factor for cropping window(prevent very small results)
* (if parameter zFactor is 1.5 => original image is 1.5x bigger than result ROI)
*/
void AutoCropVCBSCP::randomWalk(int iterations, float zoomFactor) {
	// temporary variables
	int tmpX, tmpY;
	uint32_t saliencyScore = 0;
	float saliencyRatio = 0.0f;

	// vector for saving generated candidates for cropping
	std::vector<std::array<int, 2>> candidates;

	// reverse zoom factor
	// if parameter zFactor is 1.5(original image is 1.5x bigger) => reverseZF is 0.666(2/3 of original image size)
	float reverseZFactor = 1.0f / zoomFactor;
	this->width = (int)(this->image.cols * reverseZFactor);
	this->height = (int)(this->image.rows * reverseZFactor);

	srand((unsigned int)time(NULL));

	for (int i = 0; i < iterations; i++) {
		tmpX = rand() % (this->image.cols - this->width);
		tmpY = rand() % (this->image.rows - this->height);
		if (this->salMap.cols < (tmpX + this->width)) continue;
		if (this->salMap.rows < (tmpY + this->height)) continue;

		// compute saliency energy of actual window
		saliencyScore = this->computeSaliencyEnergy(tmpX, tmpY, this->width, this->height);

		/* Content Preservation Part */
		saliencyRatio = (float)saliencyScore / (float)this->totalSaliencyEnergy;
		// there is some treshold, empirically defined
		if (saliencyRatio > 0.4f) {
			std::array<int, 2> xy = { tmpX, tmpY };
			candidates.push_back(xy);
		}
		/* Content Preservation Part End */
	}

	// init variables for Boundary simplicity cue
	uint32_t tmpScore = DEF_UINT32_MAX;
	uint32_t bestBoundaryScore = DEF_UINT32_MAX;
	int bestIndex = 0;

	for (int i = 0; i < candidates.size(); i++) {
		/* Boundary Simplicity Part */
		tmpScore = this->computeBoundarySimplicity(std::get<0>(candidates.at(i)),
												std::get<1>(candidates.at(i)),
												this->width,
												this->height);
		// finding the smallest score (pixel in boundary of objects has bigger values)
		if (tmpScore < bestBoundaryScore) {
			// save index in candidates and its score
			bestIndex = i;
			bestBoundaryScore = tmpScore;
		}
		/* Boundary Simplicity Part End */
	}

	// save coordinates of top left corner => bestIndex was saved above
	if (candidates.size() > 0) {
		this->x = std::get<0>(candidates.at(bestIndex));
		this->y = std::get<1>(candidates.at(bestIndex));
	}

	
	/*// remove Boundary Simplicity cue (Result is best of Content preservation clue) 
	uint32_t bestSalScore = 0;
	tmpScore = 0;
	for (int i = 0; i < candidates.size(); i++) {
		tmpScore = this->computeSaliencyEnergy(std::get<0>(candidates.at(i)), std::get<1>(candidates.at(i)), this->width, this->height);
		// finding the biggest score (pixel in boundary of objects has bigger values)
		if (tmpScore > bestSalScore) {
			cout << "Hey" << endl;
			bestSalScore = tmpScore;
			this->x = std::get<0>(candidates.at(i));
			this->y = std::get<1>(candidates.at(i));
		}
	}*/
}



/**
 * Method for computing saliency energy of selected ROI
 * @param x1 Horizontal coordinate of top left corner ROI
 * @param y1 Vertical coordinate of top left corner ROI
 * @param w Width of ROI
 * @param h Height of ROI
 * @return Saliency energy of defined ROI
 */
uint32_t AutoCropVCBSCP::computeSaliencyEnergy(int x1, int y1, int w, int h) {
	uint32_t score = 0;

	// right bottom corner of ROI
	int x2 = x1 + w;
	int y2 = y1 + h;

	// input condition
	if (this->salMap.cols <= x2)
		return 0;
	if (this->salMap.rows <= y2)
		return 0;

	uint8_t valPixel;
	for (int i = x1; i < x2; i++) {
		for (int j = y1; j < y2; j++) {
			// get pixel value [0,255]
			valPixel = this->salMap.data[j * salMap.cols + i];

			// increment score with pixel value [0,255]
			score += valPixel;
		}
	}

	return score;
}


/**
 * Method for computing boundary simplicity value of selected ROI
 * @param x1 Horizontal coordinate of top left corner ROI
 * @param y1 Vertical coordinate of top left corner ROI
 * @param w Width of ROI
 * @param h Height of ROI
 * @return Boundary simplicity value of defined ROI
 */
uint32_t AutoCropVCBSCP::computeBoundarySimplicity(int x1, int y1, int w, int h) {
	uint32_t score = 0;

	// right bottom corner of ROI
	int x2 = x1 + w;
	int y2 = y1 + h;

	// input condition
	if (this->gradient.cols <= x2)
		return 0;
	if (this->gradient.rows <= y2)
		return 0;


	for (int i = x1; i < x2; i++) {
		// increment score with pixel value [0,255]
		// top bound
		score += (uint8_t)this->gradient.data[y1 * this->gradient.cols + i];
		// bottom bound
		score += (uint8_t)this->gradient.data[y2 * this->gradient.cols + i];
	}
	
	for (int j = (y1 + 1); j < (y2 - 1); j++) {
		// increment score with pixel value [0,255]
		// left bound
		score += (uint8_t)this->gradient.data[j * this->gradient.cols + x1];
		// right bound
		score += (uint8_t)this->gradient.data[j * this->gradient.cols + x2];
	}

	return score;
}

/*
* Getter function for x1 position to crop (upper left corner)
*/
int AutoCropVCBSCP::getX() {
	return this->x;
}

/*
* Getter function for y1 position to crop (upper left corner)
*/
int AutoCropVCBSCP::getY() {
	return this->y;
}

/*
* Getter function for width of cropped area
*/
int AutoCropVCBSCP::getWidth() {
	return this->width;
}

/*
* Getter function for height of cropped area
*/
int AutoCropVCBSCP::getHeight() {
	return this->height;
}