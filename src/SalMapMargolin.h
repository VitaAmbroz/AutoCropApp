/**
 * Bachelor thesis: Algorithms for automatic image cropping
 * VUT FIT 2018
 * Author: Vít Ambrož (xambro15@stud.fit.vutbr.cz)
 * Supervisor: Doc. Ing. Martin Čadík, Ph. D.
 * File: SalMapMargolin.h
 * Github repository: https://github.com/VitaAmbroz/AutoCropApp
 */

#ifndef __SALMAPMARGOLIN_H__
#define __SALMAPMARGOLIN_H__

#include <numeric>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/* It is necessary to have VLFeat correctly downloaded - dependencies in CMakeLists.txt */
extern "C" {
#include "vl/generic.h"
#include "vl/slic.h"
}


class SalMapMargolin
{
public:
	// constructor
	SalMapMargolin(cv::Mat img);
	// main method for generating saliency map
	cv::Mat getSaliency(const Mat& img);
	// matrix for generated saliency map
	cv::Mat salMap;

private:
	// methods necessary for generating saliency map
	void _getSLICSegments(const Mat& img, std::vector<vl_uint32>& segmentation);
	float _getSLICVariances(Mat& grey, std::vector<vl_uint32>& segmentation, std::vector<float>& vars);
	cv::Mat _getPatternDistinct(const Mat& img, std::vector<vl_uint32>& segmentation, std::vector<float>& spxl_vars, float var_thresh);
	cv::Mat _getColourDistinct(const Mat& img, std::vector<vl_uint32>& segmentation, uint spxl_n);
	cv::Mat _getWeightMap(Mat& D);
	void addGaussian(Mat& img, uint x, uint y, float std, float weight);
	float mean(std::vector<float>& v);
	float var(std::vector<float>& v);

	// reference for original image
	cv::Mat image;
};

#endif //__SALMAPMARGOLIN_H__
