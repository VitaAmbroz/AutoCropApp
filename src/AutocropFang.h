/**
 * Bachelor thesis: Algorithms for automatic image cropping
 * VUT FIT 2018
 * Author: Vít Ambrož (xambro15@stud.fit.vutbr.cz)
 * Supervisor: Doc. Ing. Martin Čadík, Ph. D.
 * File: AutocropFang.h
 * Github repository: https://github.com/VitaAmbroz/AutoCropApp
 */

#ifndef __AUTOCROPFANG_H__
#define __AUTOCROPFANG_H__

#include <opencv2/opencv.hpp>
#include <array>
#include <algorithm>
#include "CompositionModel.h"

using namespace std;
using namespace cv;

/* number of generated ROIs for that will be computed scores of VC model and BS model */
const int CANDIDATES_COUNT = 1000;

/* weights of Visual composition and Boundary simplicity models */
const int WEIGHT_COMPOS = 5;
const int WEIGHT_BOUNDARY = 1;

/* default step when walking trough grid */
const int DEFAULT_STEP_FANG = 10;
/* minimum size of roi */
const float MAX_SCALE_FANG = 0.3f;
/* value for initializing treshold in Content preservation model */
const float SALIENCY_TRESHOLD_INIT = 0.8f;


class AutocropFang
{
public:
	// constructor
	AutocropFang(cv::Mat img, cv::Mat salMap, std::string path);
	
	// variants of cropping method
	void WHCrop(int w, int h, int hStep = DEFAULT_STEP_FANG, int vStep = DEFAULT_STEP_FANG);
	void scaleCrop(float scale, int hStep = DEFAULT_STEP_FANG, int vStep = DEFAULT_STEP_FANG);
	void WHratioCrop(int w, int h, int hStep = DEFAULT_STEP_FANG, int vStep = DEFAULT_STEP_FANG);
	void randomGridCrop();
	// getters for ROI
	int getX();
	int getY();
	int getWidth();
	int getHeight();
	// matrix for gradient image
	cv::Mat gradient;

private:
	// computes and normalizes scores of models and selects the best ROI
	void getBestCandidate(std::vector<std::array<int, 4>> candidates);
	// content preservation model methods
	bool candidateContentPreserv(int x1, int y1, int w, int h, float treshold);
	uint32_t computeSaliencyEnergy(int x1, int y1, int w, int h);
	// boundary simplicity model methods
	cv::Mat getGradient(cv::Mat img);
	float computeBoundarySimplicity(int x1, int y1, int w, int h);
	// visual composition method for computing score
	float computeVisualComposition(int x1, int y1, int w, int h);

	// ROI values
	int x;
	int y;
	int width;
	int height;
	// original image
	cv::Mat image;
	// saliency map CV_32F
	cv::Mat salMap;
	// normalized saliency map to [0,255] CV_8UC1
	cv::Mat salMapNorm;

	// total saliency energy of original image
	uint32_t totalSaliencyEnergy;
	// instance of visual composition model
	CompositionModel compos;
};

#endif //__AUTOCROPFANG_H__