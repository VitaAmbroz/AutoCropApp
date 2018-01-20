#ifndef __ATTENTIONBASED_H__
#define __ATTENTIONBASED_H__
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class AttentionBased
{
public:
	AttentionBased(cv::Mat img);

	int getX();
	int getY();
	int getWidth();
	int getHeight();
	void zoomFactorWalk(int hStep, int vStep, float from, float to, float step);
	void randomWalk(int iterations, float maxZoomFactor);
	void brutalForceZoomFactor(int hStep, int vStep, float zFactor);
	void brutalForceWH(int hStep, int vStep, int width, int height);
	void computeMaxScore(int x1, int y1, int width, int height);

private:
	int x;
	int y;
	int width;
	int height;
	double bestScore;
	cv::Mat image;
	uint8_t* pixelPtr;
};

#endif //__ATTENTIONBASED_H__
