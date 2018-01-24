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
	void brutalForceWH(int hStep, int vStep, int w, int h);
	void brutalForceZoomFactor(int hStep, int vStep, float zFactor);
	void zoomFactorWalk(int hStep, int vStep, float from, float to, float step);
	void randomZFWalk(int iterations, float maxZoomFactor);
	void randomWalk(int iterations, int minWidth, int minHeight);
	void computeMaxScore(int x1, int y1, int w, int h);

private:
	int x;
	int y;
	int width;
	int height;
	double bestScore;
	cv::Mat image;
};

#endif //__ATTENTIONBASED_H__
