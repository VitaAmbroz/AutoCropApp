#ifndef __AUTOCROPVCBSCP_H__
#define __AUTOCROPVCBSCP_H__
#include <opencv2/opencv.hpp>
#include <array>

#define DEF_UINT32_MAX (0xffffffff)

using namespace std;
using namespace cv;

class AutoCropVCBSCP
{
public:
	AutoCropVCBSCP(cv::Mat img, cv::Mat salMap);
	cv::Mat gradient;

	void randomWalk(int iterations, float zoomFactor);
	int getX();
	int getY();
	int getWidth();
	int getHeight();

private:
	cv::Mat getGradient(cv::Mat img);
	uint32_t computeSaliencyEnergy(int x1, int y1, int width, int height);
	uint32_t computeBoundarySimplicity(int x1, int y1, int width, int height);

	int x;
	int y;
	int width;
	int height;
	uint32_t totalSaliencyEnergy;
	cv::Mat image;
	cv::Mat salMap;
};

#endif //__AUTOCROPVCBSCP_H__
