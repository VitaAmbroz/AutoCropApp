/**
 * Bachelor thesis: Algorithms for automatic image cropping
 * VUT FIT 2018
 * Author: Vít Ambrož (xambro15@stud.fit.vutbr.cz)
 * Supervisor: Doc. Ing. Martin Čadík, Ph. D.
 * File: AutocropSuh.h
 * Github repository: https://github.com/VitaAmbroz/AutoCropApp
 */

#ifndef __AUTOCROPSUH_H__
#define __AUTOCROPSUH_H__

#include <array>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/* default step when walking trough grid */
const int DEFAULT_STEP_SUH = 10;

/* minimum size of roi */
const float MAX_SCALE_SUH = 0.4f;


class AutocropSuh
{
public:
	// default constructor
	AutocropSuh(cv::Mat sm);
	
	// methods of automatic cropping
	void bruteForceWH(int w, int h, int hStep = DEFAULT_STEP_SUH, int vStep = DEFAULT_STEP_SUH);
	void bruteForceScale(float scale, int hStep = DEFAULT_STEP_SUH, int vStep = DEFAULT_STEP_SUH);
	void bruteForceWHratio(int w, int h, float treshold, int hStep = DEFAULT_STEP_SUH, int vStep = DEFAULT_STEP_SUH);
	void greedyGeneral(float treshold);
	void bruteForceGeneral(float treshold);

	// getters and setters of ROI parameters
	int getX();
	int getY();
	int getWidth();
	int getHeight();

private:
	// ROI parameters
	int x;
	int y;
	int width;
	int height;
	// saliency map of original image
	cv::Mat salmap;
	// total saliency energy in whole original image
	uint32_t totalSaliency;
	// method for computing saliency energy of defined ROI
	uint32_t computeSaliency(int x1, int y1, int w, int h);
};

#endif //__AUTOCROPSUH_H__
