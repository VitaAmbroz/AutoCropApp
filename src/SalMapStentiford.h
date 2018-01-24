#ifndef __SALMAPSTENTIFORD_H__
#define __SALMAPSTENTIFORD_H__
#include <opencv2/opencv.hpp>
#include <array>
#include <cmath>

using namespace std;
using namespace cv;

class SalMapStentiford
{
public:
	SalMapStentiford(cv::Mat img);
	cv::Mat salMap;
	void generateSalMap(int m, int eps, int t, float treshold);

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
