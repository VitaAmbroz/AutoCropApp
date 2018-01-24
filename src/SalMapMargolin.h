#ifndef __SALMAPMARGOLIN_H__
#define __SALMAPMARGOLIN_H__
#include <opencv2/opencv.hpp>
#include <numeric>

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
	SalMapMargolin(cv::Mat img);
	cv::Mat salMap;
	cv::Mat getSaliency(const Mat& img);

private:
	cv::Mat image;
	void showImage(std::string title, const Mat& img);
	void addGaussian(Mat& img, uint x, uint y, float std, float weight);
	float mean(std::vector<float>& v);
	float var(std::vector<float>& v);
	void _getSLICSegments(const Mat& img, std::vector<vl_uint32>& segmentation);
	float _getSLICVariances(Mat& grey, std::vector<vl_uint32>& segmentation, std::vector<float>& vars);
	cv::Mat _getPatternDistinct(const Mat& img, std::vector<vl_uint32>& segmentation, std::vector<float>& spxl_vars, float var_thresh);
	cv::Mat _getColourDistinct(const Mat& img, std::vector<vl_uint32>& segmentation, uint spxl_n);
	cv::Mat _getWeightMap(Mat& D);
};

#endif //__SALMAPMARGOLIN_H__
