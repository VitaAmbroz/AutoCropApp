/**
 * Bachelor thesis: Algorithms for automatic image cropping
 * VUT FIT 2018
 * Author: Vít Ambrož (xambro15@stud.fit.vutbr.cz)
 * Supervisor: Doc. Ing. Martin Čadík, Ph. D.
 * File: SalMapStentiford.h
 * Github repository: https://github.com/VitaAmbroz/AutoCropApp
 */

#ifndef __SALMAPSTENTIFORD_H__
#define __SALMAPSTENTIFORD_H__

#include <array>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

// constant to get max distance for translating forkSA to forkSB
// 0 <=> generate forksB over all image X Big value <=> generate forksB close
const int TRANSLATION_DIVIDER = 10;
// default number of pixels in fork
const int DEFAULT_M = 3;
// default max distance of pixels in neighbourhood from actual pixel
const int DEFAULT_EPS = 1;
// default number of created forks
const int DEFAULT_T = 80;
// default value of threshold
const int DEFAULT_THRESHOLD = 150;


class SalMapStentiford
{
public:
	// constructor
	SalMapStentiford(cv::Mat img);
	// matrix for output saliency map
	cv::Mat salMap;
	// method for generating saliency map with default parameters
	void generateSalMap(int m = DEFAULT_M, int eps = DEFAULT_EPS, int t = DEFAULT_T, float treshold = DEFAULT_THRESHOLD);

private:
	// reference for original image
	cv::Mat originalImage;
	// input image could be possibly scaled down, it would be saved here
	cv::Mat image;
	// max distance between forkSA and forkSB in horizontal and verical direction
	int hTranslation;
	int vTranslation;

	// method for checking mismatch between two pixels
	bool mismatchPixels(int x1, int y1, int x2, int y2, float treshold);
	// methods for creating forkSA and forkSB
	std::vector<std::array<int, 2>> createForkSA(int x1, int y1, int m, int eps);
	std::vector<std::array<int, 2>> createForkSB(std::vector<std::array<int, 2>> sa, int m);
	// methods for watching over limits
	int checkMaxMinWidth(int pxX);
	int checkMaxMinHeight(int pxY);
};

#endif //__SALMAPSTENTIFORD_H__
