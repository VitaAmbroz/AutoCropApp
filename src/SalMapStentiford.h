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

// definition of constant to get max distance for translating forkSA to forkSB
// 0 <=> generate forksB over all image X Big value <=> generate forksB close
static const int TRANSLATION_DIVIDER = 10;


class SalMapStentiford
{
public:
	SalMapStentiford(cv::Mat img);
	cv::Mat salMap;
	void generateSalMap(int m, int eps, int t, float treshold); /* TODO constructor with default parameters */

private:
	cv::Mat originalImage;
	cv::Mat image;
	int hTranslation;
	int vTranslation;

	bool mismatchPixels(int x1, int y1, int x2, int y2, float treshold);
	std::vector<std::array<int, 2>> createForkSA(int x1, int y1, int m, int eps);
	std::vector<std::array<int, 2>> createForkSB(std::vector<std::array<int, 2>> sa, int m);
	int checkMaxMinWidth(int pxX);
	int checkMaxMinHeight(int pxY);
};

#endif //__SALMAPSTENTIFORD_H__
