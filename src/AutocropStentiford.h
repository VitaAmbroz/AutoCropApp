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
/* minimum size of roi */
const float MAX_SCALE_STENTIFORD = 0.3f;

class AutocropStentiford
{
public:
	AutocropStentiford(cv::Mat img);

	int getX();
	int getY();
	int getWidth();
	int getHeight();
	void brutalForceWH(int w, int h, int hStep = DEFAULT_STEP_STENTIFORD, int vStep = DEFAULT_STEP_STENTIFORD);
	void brutalForceZoomFactor(float zFactor, int hStep = DEFAULT_STEP_STENTIFORD, int vStep = DEFAULT_STEP_STENTIFORD);
	void brutalForceLimit(float maxScale, int hStep = DEFAULT_STEP_STENTIFORD, int vStep = DEFAULT_STEP_STENTIFORD);
	void zoomFactorWalk(float from, float to, float step, int hStep = DEFAULT_STEP_STENTIFORD, int vStep = DEFAULT_STEP_STENTIFORD);
	void randomZFWalk(int iterations, float maxZoomFactor);
	void randomWalk(int iterations, int minWidth, int minHeight);
	void computeMaxScore(int x1, int y1, int w, int h);

	void runTest();

private:
	int x;
	int y;
	int width;
	int height;
	double bestScore;
	cv::Mat image;
};

#endif //__AUTOCROPSTENTIFORD_H__