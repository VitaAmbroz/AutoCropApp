/**
 * Bachelor thesis: Algorithms for automatic image cropping
 * VUT FIT 2018
 * Author: Vít Ambrož (xambro15@stud.fit.vutbr.cz)
 * Supervisor: Doc. Ing. Martin Čadík, Ph. D.
 * File: AutocropSuh.cpp
 * Github repository: https://github.com/VitaAmbroz/AutoCropApp
 */

#include "AutocropSuh.h"

AutocropSuh::AutocropSuh(cv::Mat sm) {
	this->x = 0;
	this->y = 0;
	this->width = sm.cols;
	this->height = sm.rows;
	this->salmap = sm;
	// compute saliency value of full ROI
	this->totalSaliency = this->computeSaliency(0, 0, this->width, this->height); // sum(this->salmap)[0];
}


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

	for (int xx = 0; xx < endColumn; xx += hStep) {
		for (int yy = 0; yy < endRow; yy += vStep) {
			for (int j = 0; j < 10; j++) { // generate 10 ROIs at each location in grid
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

	for (int xx = 0; xx < endColumn; xx += hStep) {
		for (int yy = 0; yy < endRow; yy += vStep) {
			uint32_t salValue = this->computeSaliency(xx, yy, w, h);
			double ratio = (double)salValue / (double)this->totalSaliency;

			if (ratio > bestRatio) {
				bestRatio = ratio;
				this->x = xx;
				this->y = yy;
			}
		}
	}
}


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

	for (int xx = 0; xx < endColumn; xx += hStep) {
		for (int yy = 0; yy < endRow; yy += vStep) {
			uint32_t salValue = this->computeSaliency(xx, yy, this->width, this->height);
			double ratio = (double)salValue / (double)this->totalSaliency;

			if (ratio > bestRatio) {
				bestRatio = ratio;
				this->x = xx;
				this->y = yy;
			}
		}
	}
}


void AutocropSuh::bruteForceWHratio(int w, int h, float treshold, int hStep, int vStep) {
	if (w < 1 || h < 1 || treshold <= 0.0f || treshold >= 1.0f || hStep < 1 || vStep < 1)
		return;

	const int SIZE_INCREMENT = (w > h) ? (int)(0.05f * this->salmap.cols) : (int)(0.05f * this->salmap.rows);

	int tmpW, tmpH;
	if (w > h) {
		tmpW = (int)(MAX_SCALE_SUH * this->salmap.cols);
		tmpH = (int)(tmpW * h / w);
	}
	else {
		tmpH = (int)(MAX_SCALE_SUH * this->salmap.rows);
		tmpW = (int)(tmpH * w / h);
	}

	while ((tmpW < this->salmap.cols) && (tmpH < this->salmap.rows)) {
		// values for end of loop
		int endColumn = this->salmap.cols - tmpW;
		int endRow = this->salmap.rows - tmpH;

		for (int xx = 0; xx < endColumn; xx += hStep) {
			for (int yy = 0; yy < endRow; yy += vStep) {
				uint32_t salValue = this->computeSaliency(xx, yy, tmpW, tmpH);
				double ratio = (double)salValue / (double)this->totalSaliency;

				if (ratio > treshold) {
					this->x = xx;
					this->y = yy;
					this->width = tmpW;
					this->height = tmpH;
					return;
				}
			}
		}

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


void greedyGeneral(float treshold) {

}


uint32_t AutocropSuh::computeSaliency(int x1, int y1, int w, int h) {
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