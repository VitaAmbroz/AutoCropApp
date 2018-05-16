/**
 * Bachelor thesis: Algorithms for automatic image cropping
 * VUT FIT 2018
 * Author: Vít Ambrož (xambro15@stud.fit.vutbr.cz)
 * Supervisor: Doc. Ing. Martin Čadík, Ph. D.
 * File: AutocropStentiford.h
 * Github repository: https://github.com/VitaAmbroz/AutoCropApp
 */

#ifndef __AUTOCROPSTENTIFORD_H__
#define __AUTOCROPSTENTIFORD_H__

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/* default step when walking through grid */
const int DEFAULT_STEP_STENTIFORD = 10;
/* number of iterations in methods using random generator */
const int ITERATIONS = 4000;

class AutocropStentiford
{
public:
	// constructor
	AutocropStentiford(cv::Mat sm);
	// different methods of automatic cropping
	void brutalForceWH(int w, int h, int hStep = DEFAULT_STEP_STENTIFORD, int vStep = DEFAULT_STEP_STENTIFORD);
	void brutalForceZoomFactor(float zFactor, int hStep = DEFAULT_STEP_STENTIFORD, int vStep = DEFAULT_STEP_STENTIFORD);
	void zoomFactorWalk(float from, float to, float step, int hStep = DEFAULT_STEP_STENTIFORD, int vStep = DEFAULT_STEP_STENTIFORD);
	void randomWHratio(int w, int h, float maxZoomFactor);
	void randomZFWalk(float maxZoomFactor);
	void randomWalk(int minWidth, int minHeight);
	// method for computing the best average pixel attention score
	void computeMaxScore(int x1, int y1, int w, int h);

	// getters and setters
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
	// value of actual best score of ROI with highest attention measure
	double bestScore;
	// saliency map from original image
	cv::Mat salMap;
};

#endif //__AUTOCROPSTENTIFORD_H__