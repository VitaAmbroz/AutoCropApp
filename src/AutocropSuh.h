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
#include <opencv2/opencv.hpp>
#include "SalMapStentiford.h"

using namespace std;
using namespace cv;

/* default step when walking trough grid */
const int DEFAULT_STEP_SUH = 5;

/* minimum size of roi */
const float MAX_SCALE_SUH = 0.25f;


class AutocropSuh
{
public:
	AutocropSuh(cv::Mat sm);
	
	int getX();
	int getY();
	int getWidth();
	int getHeight();
	void bruteForceGeneral(float treshold);
	void bruteForceWH(int w, int h, int hStep = DEFAULT_STEP_SUH, int vStep = DEFAULT_STEP_SUH);
	void bruteForceScale(float scale, int hStep = DEFAULT_STEP_SUH, int vStep = DEFAULT_STEP_SUH);
	void bruteForceWHratio(int w, int h, float treshold, int hStep = DEFAULT_STEP_SUH, int vStep = DEFAULT_STEP_SUH);
	void greedyGeneral(float treshold);

private:
	int x;
	int y;
	int width;
	int height;
	cv::Mat salmap;
	uint32_t totalSaliency;

	uint32_t computeSaliency(int x1, int y1, int w, int h);
};

#endif //__AUTOCROPSUH_H__
