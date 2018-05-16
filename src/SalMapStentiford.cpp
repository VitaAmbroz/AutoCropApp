/**
 * Bachelor thesis: Algorithms for automatic image cropping
 * VUT FIT 2018
 * Author: Vít Ambrož (xambro15@stud.fit.vutbr.cz)
 * Supervisor: Doc. Ing. Martin Čadík, Ph. D.
 * File: SalMapStentiford.cpp
 * Github repository: https://github.com/VitaAmbroz/AutoCropApp
 */

#include "SalMapStentiford.h"

/**
 * Default constructor
 * @param img Original image
 */
SalMapStentiford::SalMapStentiford(cv::Mat img) {
	this->originalImage = img;
	this->salMap = Mat();

	this->hTranslation = (TRANSLATION_DIVIDER != 0) ? (int)(img.cols / TRANSLATION_DIVIDER) : img.cols;
	this->vTranslation = (TRANSLATION_DIVIDER != 0) ? (int)(img.rows / TRANSLATION_DIVIDER) : img.rows;
}


/**
* Method for generating saliency map of input image, it is saved to matrix salMap
* @param m Number of pixels in each fork
* @param eps Definiton for neighbourhood of current pixels = max distance of other pixels 
* @param t Number of generated forks
* @param treshold Value that defines if fork A mismatches fork B 
*/
void SalMapStentiford::generateSalMap(int m, int eps, int t, float treshold) {
	srand((unsigned int)time(NULL));

	// big images would be scaled down - max 400px width or height
	const float maxSize = 400.f;
	double scale = maxSize / max(this->originalImage.cols, this->originalImage.rows);

	if (scale >= 1.f) {	// dont resize
		this->image = this->originalImage;
	}
	else { // image is large, resize it to max 800px of width or height
		// use INTER_AREA to resampling using pixel area relation
		cv::resize(this->originalImage, this->image, Size(), scale, scale, cv::INTER_AREA);
	}


	// Mat for saving saliency values each pixel
	cv::Mat salmap_scaled = Mat(this->image.rows, this->image.cols, CV_8UC1);

	// loop and computation for every single pixel of original image
#pragma omp parallel for
	for (int xx = 0; xx < this->image.cols; xx++) {
		for (int yy = 0; yy < this->image.rows; yy++) {
			int pxAttentionScore = 0;
			bool forkMisMatch = false;
		
			for (int i = 0; i < t; i++) {
				std::vector<std::array<int, 2>> forkSA = this->createForkSA(xx, yy, m, eps);
				std::vector<std::array<int, 2>> forkSB = this->createForkSB(forkSA, m);

				forkMisMatch = false;
				for (int j = 0; j < m; j++) {
					// check mismatch of forks
					if (this->mismatchPixels(std::get<0>(forkSA.at(j)),
											std::get<1>(forkSA.at(j)),
											std::get<0>(forkSB.at(j)),
											std::get<1>(forkSB.at(j)),
											treshold)) {

						forkMisMatch = true; // mismatch detected -> score will be incremented
						break;
					}
				}

				// increment score if forkSA mismatches forkSB
				if (forkMisMatch) {
					pxAttentionScore += 1;
				}
			}

			// max value of pixel
			if (pxAttentionScore > 255)
				pxAttentionScore = 255;

			// save attention score to output saliency map
			salmap_scaled.data[yy * this->image.cols + xx] = pxAttentionScore;
		}
	}

	// If image has been scaled down, now scale it back
	if (this->originalImage.cols > (int)maxSize || this->originalImage.rows > (int)maxSize) {
		// Scale back to original size for further processing
		cv::resize(salmap_scaled, this->salMap, this->originalImage.size());
	}
	else {  // image has not been scaled down, no need to scale back
		this->salMap = salmap_scaled;
	}

	// normalize it in range [0,255]
	cv::normalize(this->salMap, this->salMap, 0, 255, NORM_MINMAX, CV_8UC1);
}


/**
* Method that detects mismatching between two pixels
* @param x1 Coordination X of first pixel
* @param y1 Coordination Y of first pixel
* @param x2 Coordination X of second pixel
* @param y2 Coordination Y of second pixel
* @param treshold Value that defines if pixel 1 mismatches pixel 2
* @return true if mismatching is detected, else returns false
*/
bool SalMapStentiford::mismatchPixels(int x1, int y1, int x2, int y2, float treshold) {
	// number of channels
	int channels = this->image.channels();

	// L1 norm
	//double Fxy = abs(this->image.data[channels * (this->image.cols * y1 + x1) + 0] - this->image.data[channels * (this->image.cols * y2 + x2) + 0]) +
	//	abs(this->image.data[channels * (this->image.cols * y1 + x1) + 1] - this->image.data[channels * (this->image.cols * y2 + x2) + 1]) +
	//	abs(this->image.data[channels * (this->image.cols * y1 + x1) + 2] - this->image.data[channels * (this->image.cols * y2 + x2) + 2]);
	
	// L2 norm
	double Fxy = sqrt(pow(this->image.data[channels * (this->image.cols * y1 + x1) + 0] - this->image.data[channels * (this->image.cols * y2 + x2) + 0], 2) +
		pow(this->image.data[channels * (this->image.cols * y1 + x1) + 1] - this->image.data[channels * (this->image.cols * y2 + x2) + 1], 2) +
		pow(this->image.data[channels * (this->image.cols * y1 + x1) + 2] - this->image.data[channels * (this->image.cols * y2 + x2) + 2], 2));

	// Colour Average
	/*double F1 = this->image.data[channels * (this->image.cols * y1 + x1) + 0] +
		this->image.data[channels * (this->image.cols * y1 + x1) + 1] +
		this->image.data[channels * (this->image.cols * y1 + x1) + 2];
	F1 /= 3;

	double F2 = this->image.data[channels * (this->image.cols * y2 + x2) + 0] +
		this->image.data[channels * (this->image.cols * y2 + x2) + 1] +
		this->image.data[channels * (this->image.cols * y2 + x2) + 2];
	F2 /= 3;

	double Fxy = abs(F1 - F2);*/

	// empiric grayscale equation
	// double F1 = 0.11 * this->image.data[channels * (this->image.cols * y1 + x1) + 0] +	// blue
	// 	0.59 * this->image.data[channels * (this->image.cols * y1 + x1) + 1] +		// green
	// 	0.3 * this->image.data[channels * (this->image.cols * y1 + x1) + 2];		// red

	// double F2 = 0.11 * this->image.data[channels * (this->image.cols * y2 + x2) + 0] +	// blue
	// 	0.59 * this->image.data[channels * (this->image.cols * y2 + x2) + 1] +		// green
	// 	0.3 * this->image.data[channels * (this->image.cols * y2 + x2) + 2];		// red

	// double Fxy = abs(F1 - F2);

	if (Fxy > treshold)
		return true;

	return false;
}


/**
* Method that creates fork SA of pixels
* @param x1 Coordination X of pixel
* @param y1 Coordination Y of pixel
* @param m Number of pixels in fork
* @param eps Neighbouring distance from pixel
* @return Vector that represents fork of pixels
*/
std::vector<std::array<int, 2>> SalMapStentiford::createForkSA(int x1, int y1, int m, int eps) {
	std::vector<std::array<int, 2>> forkSA;

	// limits for neighbourhood
	int minX = x1 - eps;
	int maxX = x1 + eps;
	int minY = y1 - eps;
	int maxY = y1 + eps;
	minX = (minX >= 0) ? minX : 0;
	maxX = (maxX < this->image.cols) ? maxX : (this->image.cols - 1);
	minY = (minY >= 0) ? minY : 0;
	maxY = (maxY < this->image.rows) ? maxY : (this->image.rows - 1);

	for (int i = 0; i < m; i++) {
		// generate random coordinates of defined neighbourhood
		int newX = minX + (rand() % (1 + maxX - minX));
		int newY = minY + (rand() % (1 + maxY - minY));

		// save coordinates of pixel to forkSA
		std::array<int, 2> xy = {newX, newY};
		forkSA.push_back(xy);
	}

	return forkSA;
}


/**
* Method that creates fork SB of pixels(translated version of forkSB)
* @param sa Vector that represents forkSB
* @param m Number of pixels in fork
* @return Vector that represents fork SB of pixels
*/
std::vector<std::array<int, 2>> SalMapStentiford::createForkSB(std::vector<std::array<int, 2>> sa, int m) {
	std::vector<std::array<int, 2>> forkSB;

	// translation lengths in horizontal and vertical coordinates
	int deltaX = 1 + (rand() % this->hTranslation);
	int deltaY = 1 + (rand() % this->vTranslation);

	// it can be translated in all directions
	int minusChanceX = rand() % 100;
	int minusChanceY = rand() % 100;
	if (minusChanceX < 50) deltaX *= -1;
	if (minusChanceY < 50) deltaY *= -1;

	for (int i = 0; i < m; i++) {
		// save coordinates of pixel to forkSB
		std::array<int, 2> xy = { this->checkMaxMinWidth(std::get<0>(sa.at(i)) + deltaX),
								this->checkMaxMinHeight(std::get<1>(sa.at(i)) + deltaY) };
		forkSB.push_back(xy);
	}

	return forkSB;
}

/**
* Method for checking if horizontal position of pixel is not out of bounds,
* if so then return minimal or maximal value.
* @param pxX value of horizontal position of the current pixel
* @return correct value that is not out of bounds
*/
int SalMapStentiford::checkMaxMinWidth(int pxX) {
	int result = pxX;

	if (pxX < 0) result = 0;
	else if (pxX >= this->image.cols) result = this->image.cols - 1;

	return result;
}


/**
* Method for checking if vertical position of pixel is not out of bounds,
* if so then return minimal or maximal value.
* @param pxX value of vertical position of the current pixel
* @return correct value that is not out of bounds
*/
int SalMapStentiford::checkMaxMinHeight(int pxY) {
	int result = pxY;

	if (pxY < 0) result = 0;
	else if (pxY >= this->image.rows) result = this->image.rows - 1;

	return result;
}