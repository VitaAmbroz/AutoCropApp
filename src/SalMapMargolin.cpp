/**
 * Bachelor thesis: Algorithms for automatic image cropping
 * VUT FIT 2018
 * Author: Vít Ambrož (xambro15@stud.fit.vutbr.cz)
 * Supervisor: Doc. Ing. Martin Čadík, Ph. D.
 * File: SalMapMargolin.cpp
 * Github repository: https://github.com/VitaAmbroz/AutoCropApp
 */

/*
 * This implementation of saliency map (Margolin, R.; Tal, A.; Zelnik-Manor, L.: What Makes a Patch Distinct?, 2013)
 * was edited and taken from https://github.com/swook/autocrop/tree/master/src/saliency
 * and uses functions from open source library VLFeat.
 */

#include "SalMapMargolin.h"

using namespace std;
using namespace cv;


/**
 * Constructor
 * @param img Original image
 */
SalMapMargolin::SalMapMargolin(cv::Mat img) {
	this->image = img;
	this->salMap = this->getSaliency(img);
}


/**
* Generates a saliency map using a method from Margolin et al. (2013)
*
* 1) Acquire pattern distinctiveness map
* 2) Acquire colour distinctiveness map
* 3) Calculate pixelwise multiplication of the two maps
*/
cv::Mat SalMapMargolin::getSaliency(const cv::Mat& img)
{
	cv::Mat img_BGR_1;

	uint H = img.rows,
		W = img.cols,
		HW = H * W;

	const float maxSize = 600.f;
	// Scale image to have not more than maxSize pixels on its larger
	// dimension
	float scale = (float)max(H, W) / maxSize;
	W = (int)(W / scale);
	H = (int)(H / scale);
	cv::resize(img, img_BGR_1, Size(W, H));
	

	cv::Mat img_BGR_2, img_BGR_4;
	cv::resize(img_BGR_1, img_BGR_2, Size(W / 2, H / 2));
	cv::resize(img_BGR_1, img_BGR_4, Size(W / 4, H / 4));

	// Get grayscale image to work on
	auto img_grey_1 = cv::Mat(img_BGR_1.size(), img_BGR_1.type());
	auto img_grey_2 = cv::Mat(img_BGR_2.size(), img_BGR_2.type());
	auto img_grey_4 = cv::Mat(img_BGR_4.size(), img_BGR_4.type());
	cv::cvtColor(img_BGR_1, img_grey_1, CV_BGR2GRAY);
	cv::cvtColor(img_BGR_2, img_grey_2, CV_BGR2GRAY);
	cv::cvtColor(img_BGR_4, img_grey_4, CV_BGR2GRAY);
	auto img_lab_1 = cv::Mat(img_BGR_1.size(), img_BGR_1.type());
	auto img_lab_2 = cv::Mat(img_BGR_2.size(), img_BGR_2.type());
	auto img_lab_4 = cv::Mat(img_BGR_4.size(), img_BGR_4.type());
	cv::cvtColor(img_BGR_1, img_lab_1, CV_BGR2Lab);
	cv::cvtColor(img_BGR_2, img_lab_2, CV_BGR2Lab);
	cv::cvtColor(img_BGR_4, img_lab_4, CV_BGR2Lab);

	// Get SLIC superpixels
	auto segmentation_1 = std::vector<vl_uint32>(H*W);
	auto segmentation_2 = std::vector<vl_uint32>(H*W / 4);
	auto segmentation_4 = std::vector<vl_uint32>(H*W / 16);
	this->_getSLICSegments(img_lab_2, segmentation_2);
	this->_getSLICSegments(img_lab_4, segmentation_4);
	this->_getSLICSegments(img_lab_1, segmentation_1); // Out-of-order for vis purposes

	// Calculate variance of super pixels
	auto spxl_n_1 = std::accumulate(segmentation_1.begin(), segmentation_1.end(), 0,
		[&](vl_uint32 b, vl_uint32 n) { return n > b ? n : b; }) + 1;
	auto spxl_n_2 = std::accumulate(segmentation_2.begin(), segmentation_2.end(), 0, 
		[&](vl_uint32 b, vl_uint32 n) { return n > b ? n : b; }) + 1;
	auto spxl_n_4 = std::accumulate(segmentation_4.begin(), segmentation_4.end(), 0,
		[&](vl_uint32 b, vl_uint32 n) { return n > b ? n : b; }) + 1;
	auto spxl_vars_1 = std::vector<float>(spxl_n_1);
	auto spxl_vars_2 = std::vector<float>(spxl_n_2);
	auto spxl_vars_4 = std::vector<float>(spxl_n_4);
	auto var_thresh_1 = this->_getSLICVariances(img_grey_1, segmentation_1, spxl_vars_1);
	auto var_thresh_2 = this->_getSLICVariances(img_grey_2, segmentation_2, spxl_vars_2);
	auto var_thresh_4 = this->_getSLICVariances(img_grey_4, segmentation_4, spxl_vars_4);

	// Compute pattern distinctiveness maps
	cv::Mat patternD_1 = this->_getPatternDistinct(img_grey_1, segmentation_1, spxl_vars_1, 200); //var_thresh_1);
	cv::Mat patternD_2 = this->_getPatternDistinct(img_grey_2, segmentation_2, spxl_vars_2, 200); //var_thresh_2);
	cv::Mat patternD_4 = this->_getPatternDistinct(img_grey_4, segmentation_4, spxl_vars_4, 200); //var_thresh_4);
	cv::Mat patternD_2_, patternD_4_;
	cv::resize(patternD_2, patternD_2_, patternD_1.size());
	cv::resize(patternD_4, patternD_4_, patternD_1.size());
	cv::Mat patternD = (patternD_1 + patternD_2_ + patternD_4_) / 3;

	// Compute colour distinctiveness maps
	cv::Mat colourD_1 = this->_getColourDistinct(img_lab_1, segmentation_1, spxl_n_1);
	cv::Mat colourD_2 = this->_getColourDistinct(img_lab_2, segmentation_2, spxl_n_2);
	cv::Mat colourD_4 = this->_getColourDistinct(img_lab_4, segmentation_4, spxl_n_4);
	cv::Mat colourD_2_, colourD_4_;
	cv::resize(colourD_2, colourD_2_, colourD_1.size());
	cv::resize(colourD_4, colourD_4_, colourD_1.size());
	cv::Mat colourD = (colourD_1 + colourD_2_ + colourD_4_) / 3;

	// Calculate distinctiveness map from pattern and colour distinctiveness
	cv::Mat D = colourD.mul(patternD);

	// Compute Gaussian weight map to highlight clusters
	cv::Mat G = this->_getWeightMap(D);

	// Final Saliency Map
	cv::Mat out;
	cv::normalize(D.mul(G), out, 0.f, 1.f, NORM_MINMAX, CV_32FC1);

	cv::Mat element = cv::getStructuringElement(MORPH_ELLIPSE, Size(5, 5), Point(2, 2));
	cv::dilate(out, out, element);
	cv::GaussianBlur(out, out, Size(11, 11), 50.f);

	// Scale back to original size for further processing
	cv::Mat out_scaled = Mat();
	cv::resize(out, out_scaled, img.size());
	std::swap(out, out_scaled);

	// Normalize it to 0-255 values
	/*Mat out_norm;
	normalize(out, out_norm, 0, 255, NORM_MINMAX, CV_8UC1);*/

	return out;
}

/**
* Calculates SLIC segmentation for a given LAB image
*/
void SalMapMargolin::_getSLICSegments(const Mat& img, std::vector<vl_uint32>& segmentation)
{
	uint H = img.rows,
		W = img.cols,
		HW = H * W;

	// Convert format from LABLAB to LLAABB (for vlfeat)
	auto img_vl = new float[HW * 3];
	auto img_ = img.ptr<Vec3b>(0);
	for (uint j = 0; j < H; j++) {
		for (uint i = 0; i < W; i++) {
			img_vl[j * W + i] = img_[j * W + i][0];
			img_vl[j * W + i + HW] = img_[j * W + i][1];
			img_vl[j * W + i + HW * 2] = img_[j * W + i][2];
		}
	}

	// Run SLIC code from vlfeat
	vl_size regionSize = 50, minRegionSize = 35;
	vl_slic_segment(segmentation.data(), img_vl, W, H, img.channels(),
		regionSize, 800, minRegionSize);

	//return; // Skip visualisation. Comment out to tune parameters.

	// Visualise segmentation
	Mat vis;
	cvtColor(img, vis, CV_Lab2BGR);
	int** labels = new int*[H];
	for (uint j = 0; j < H; j++) {
		labels[j] = new int[W];
		for (uint i = 0; i < W; i++)
			labels[j][i] = (int)segmentation[j*W + i];
	}

	int label, labelTop, labelBottom, labelLeft, labelRight;
	for (uint j = 1; j < H - 1; j++) {
		for (uint i = 1; i < W - 1; i++) {
			label = labels[j][i];
			labelTop = labels[j - 1][i];
			labelBottom = labels[j + 1][i];
			labelLeft = labels[j][i - 1];
			labelRight = labels[j][i + 1];
			if (label != labelTop || label != labelBottom ||
				label != labelLeft || label != labelRight) {
				vis.at<Vec3b>(j, i)[0] = 0;
				vis.at<Vec3b>(j, i)[1] = 0;
				vis.at<Vec3b>(j, i)[2] = 255;
			}
		}
	}
}

float SalMapMargolin::_getSLICVariances(Mat& grey, std::vector<vl_uint32>& segmentation, std::vector<float>& vars)
{
	uint n = (uint)vars.size();
	uint HW = grey.cols * grey.rows;

	// 1. Aggregate pixels by super pixel
	auto spxl_vals = std::vector<std::vector<float>>(n);
	for (uint i = 0; i < n; i++) {
		spxl_vals[i] = std::vector<float>(0);
		spxl_vals[i].reserve(20);
	}

	vl_uint32 spxl_id;
	float     spxl_val;
	for (uint i = 0; i < HW; i++) {
		spxl_id = segmentation[i];
		spxl_val = (float)grey.ptr<uchar>(0)[i];
		spxl_vals[spxl_id].push_back(spxl_val);
	}

	// 2. Calculate variance of group of pixels
	for (uint i = 0; i < n; i++)
		vars[i] = var(spxl_vals[i]);

	// 3. Calculate variance threshold (25% with highest variance)
	auto vars_sorted = vars;
	std::sort(vars_sorted.begin(), vars_sorted.end());
	return vars_sorted[n - n / 4];
}





/**
* Generates a pattern distinctiveness map
*
* 1) [Divide image into 9x9 patches]
* 2) Perform PCA
* 3) Project each patch into PCA space
* 4) Take L1-norm and store to map
*/
Mat SalMapMargolin::_getPatternDistinct(const Mat& img, std::vector<vl_uint32>& segmentation, std::vector<float>& spxl_vars, float var_thresh)
{
	const float _9 = 1.f / 9.f;
	
	uint H = img.rows,
		W = img.cols,
		Y = H - 1,    // Limit of y indexing
		X = W - 1,
		IH = H - 2,    // Inner width (sans 1-pixel border)
		IW = W - 2;    // Inner height

	const uchar* row_p = img.ptr<uchar>(0); // Pixel values of i-1th row
	const uchar* row = img.ptr<uchar>(1); // Pixel values of ith row
	const uchar* row_n;                     // Pixel values of i+1th row

	/******************************/
	/* Create list of 9x9 patches */
	/******************************/
	auto _patches = std::vector<uchar>(0);
	_patches.reserve(X * Y * 9);

	// Patches in superpixels with var above var_thresh
	auto _distpatches = std::vector<uchar>(0);
	_distpatches.reserve(X * Y * 9);

	// Iterate over all inner pixels (patches) with variance above threshold
	uint i = 0, spxl_i;
	float p1, p2, p3, p4, p5, p6, p7, p8, p9, m;
	for (uint y = 1; y < Y; y++) {
		row_n = img.ptr<uchar>(y + 1);
		for (uint x = 1; x < X; x++) {
			p1 = row_p[x - 1]; p2 = row_p[x]; p3 = row_p[x + 1];
			p4 = row[x - 1]; p5 = row[x]; p6 = row[x + 1];
			p7 = row_n[x - 1]; p8 = row_n[x]; p9 = row_n[x + 1];
			m = _9 * (p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9);

			spxl_i = segmentation[y*X + x];
			// TODO: "To disregard lighting effects we a-priori
			//        subtract from each patch its mean value."
			if (spxl_vars[spxl_i] > var_thresh)
			{
				_distpatches.push_back((uchar)(p1 - m));
				_distpatches.push_back((uchar)(p2 - m));
				_distpatches.push_back((uchar)(p3 - m));
				_distpatches.push_back((uchar)(p4 - m));
				_distpatches.push_back((uchar)(p5 - m));
				_distpatches.push_back((uchar)(p6 - m));
				_distpatches.push_back((uchar)(p7 - m));
				_distpatches.push_back((uchar)(p8 - m));
				_distpatches.push_back((uchar)(p9 - m));
				i++;
			}

			_patches.push_back((uchar)(p1));
			_patches.push_back((uchar)(p2));
			_patches.push_back((uchar)(p3));
			_patches.push_back((uchar)(p4));
			_patches.push_back((uchar)(p5));
			_patches.push_back((uchar)(p6));
			_patches.push_back((uchar)(p7));
			_patches.push_back((uchar)(p8));
			_patches.push_back((uchar)(p9));
		}
		row_p = row;
		row = row_n;
	}
	_distpatches.shrink_to_fit();
	auto distpatches = Mat(i, 9, CV_8U, _distpatches.data());
	auto patches = Mat(X*Y, 9, CV_8U, _patches.data());
	///printf("%.1f%% of patches considered distinct\n", 100.f * (float)i / (float)(X*Y));


	/*******/
	/* PCA */
	/*******/
	auto pca = PCA(distpatches, cv::Mat(), CV_PCA_DATA_AS_ROW);
	
	cv::Mat pca_pos = cv::Mat::zeros(IH * IW, 9, CV_32F); // Coordinates in PCA space
	cv::Mat pca_norm = cv::Mat::zeros(IH * IW, 1, CV_32F); // L1 norm of pca_pos

	pca.project(patches, pca_pos); // Project patches into PCA space
	cv::reduce(abs(pca_pos), pca_norm, 1, CV_REDUCE_SUM); // Calc L1 norm

	// Pad with 1-pixel thick black border
	cv::Mat out_inner = cv::Mat(IH, IW, CV_32F, pca_norm.ptr<float>(0), 0);
	cv::Mat out;
	cv::copyMakeBorder(out_inner, out, 1, 1, 1, 1, BORDER_CONSTANT, 0);


	/*******************/
	/* Post-processing */
	/*******************/

	// Dilate-then-erode to close holes
	cv::Mat out_closed;
	cv::morphologyEx(out, out_closed, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));

	// Normalise then return
	cv::Mat out_norm;
	cv::normalize(out_closed, out_norm, 0.f, 1.f, NORM_MINMAX);
	return out_norm;
}

/**
* Generates a colour distinctiveness map
*
* 1) Calculate average colour per SLIC region
* 2) Calculate sum of euclidean distance between colours
*/
cv::Mat SalMapMargolin::_getColourDistinct(const cv::Mat& img, std::vector<vl_uint32>& segmentation, uint spxl_n)
{
	uint H = img.rows,
		W = img.cols,
		HW = H * W;

	// 1. Aggregate colours of regions
	auto spxl_cols = std::vector<Vec3f>(spxl_n);
	auto spxl_cnts = std::vector<uint>(spxl_n);
	// Allocate
	for (uint i = 0; i < spxl_n; i++)
		spxl_cols[i] = Vec3f();

	// Aggregate Lab colour values
	for (uint idx = 0, j = 0; j < H; j++)
		for (uint i = 0; i < W; i++)
		{
			idx = segmentation[j*W + i];
			spxl_cols[idx][0] += (float)img.ptr<Vec3b>(j)[i][0];
			spxl_cols[idx][1] += (float)img.ptr<Vec3b>(j)[i][1];
			spxl_cols[idx][2] += (float)img.ptr<Vec3b>(j)[i][2];
			spxl_cnts[idx]++;
		}

	// Divide by no. of pixels
	for (uint i = 0; i < spxl_n; i++)
	{
		spxl_cols[i][0] /= spxl_cnts[i];
		spxl_cols[i][1] /= spxl_cnts[i];
		spxl_cols[i][2] /= spxl_cnts[i];
	}

	// 2. Aggregate colour distances
	auto spxl_dist = std::vector<float>(spxl_n);
	float dist, weight;
	for (uint i1 = 0; i1 < spxl_n; i1++)
	{
		if (spxl_cnts[i1] == 0) continue;
		dist = 0.f;
		for (uint i2 = i1 + 1; i2 < spxl_n; i2++) {
			if (spxl_cnts[i2] == 0) continue;

			weight = (float)norm(spxl_cols[i1] - spxl_cols[i2]);

			dist += weight;
			spxl_dist[i2] += weight;
		}
		spxl_dist[i1] += dist;
	}

	// 3. Assign distance value to output colour distinctiveness map
	auto out = cv::Mat(img.size(), CV_32F);
	for (uint idx = 0, j = 0; j < H; j++)
		for (uint i = 0; i < W; i++)
		{
			idx = segmentation[j*W + i];
			out.ptr<float>(j)[i] = spxl_dist[idx];
		}

	// Normalise
	cv::Mat out_norm;
	cv::normalize(out, out_norm, 0.f, 1.f, NORM_MINMAX);
	return out_norm;
}

/**
* Generates a Gaussian weight map
*
* 1) Threshold given distinctiveness map with thresholds in 0:0.1:1
* 2) Compute centre of mass
* 3) Place Gaussian with standard deviation 1000 at CoM
*    (Weight according to threshold)
*/
cv::Mat SalMapMargolin::_getWeightMap(Mat& D)
{
	cv::Mat out = cv::Mat::zeros(D.size(), CV_32F);

	float thresh = 0.f, v, M;
	cv::Vec2f CoM;
	for (int i = 0; i < 10; i++)
	{
		M = 0.f;
		CoM[0] = 0.f;
		CoM[1] = 0.f;
		for (int y = 0; y < out.rows; y++)
			for (int x = 0; x < out.cols; x++)
			{
				v = D.ptr<float>(y)[x];
				if (v > thresh)
				{
					CoM[0] += v * (float)x;
					CoM[1] += v * (float)y;
					M += v;
				}
			}

		this->addGaussian(out, (uint)round(CoM[0] / M), (uint)round(CoM[1] / M), 10000, thresh);
		thresh += 0.1f;
	}

	// Add centre prior
	this->addGaussian(out, out.cols / 2, out.rows / 2, 10000, 5);

	// Normalise
	cv::Mat out_norm;
	cv::normalize(out, out_norm, 0.f, 1.f, NORM_MINMAX);
	return out_norm;
}


void SalMapMargolin::addGaussian(Mat& img, uint x, uint y, float std, float weight)
{
	int H = img.rows,
		W = img.cols,
		HW = H * W;

	const float PI = 3.14f;

	const float a = 1.f / (std * std * 2.f * PI);
	const float b = -0.5f / (std * std);

	auto done = std::vector<bool>(HW);  // Mark if calculation done
	float dy2, dx2, g;
	int dy, dx, j, i, j_, i_; // Mirrored pixels to check
	uint ji_;

	for (j = 0; j < H; j++)
	{
		dy = abs(j - (int)y);
		dy2 = (float)(dy * dy);
		for (i = 0; i < W; i++)
		{
			if (done[j * W + i]) continue;

			dx = abs(i - (int)x);
			dx2 = (float)(dx * dx);
			g = weight * a * exp(b * (dx2 + dy2));

			j_ = y + dy;
			i_ = x + dx;
			ji_ = j_ * W + i_;
			if (j_ < H && j_ > -1 && i_ < W && i_ > -1 && !done[ji_]) {
				img.ptr<float>(j_)[i_] += g;
				done[ji_] = true;
			}

			j_ = y - dy;
			i_ = x + dx;
			ji_ = j_ * W + i_;
			if (j_ < H && j_ > -1 && i_ < W && i_ > -1 && !done[ji_]) {
				img.ptr<float>(j_)[i_] += g;
				done[ji_] = true;
			}

			j_ = y + dy;
			i_ = x - dx;
			ji_ = j_ * W + i_;
			if (j_ < H && j_ > -1 && i_ < W && i_ > -1 && !done[ji_]) {
				img.ptr<float>(j_)[i_] += g;
				done[ji_] = true;
			}

			j_ = y - dy;
			i_ = x - dx;
			ji_ = j_ * W + i_;
			if (j_ < H && j_ > -1 && i_ < W && i_ > -1 && !done[ji_]) {
				img.ptr<float>(j_)[i_] += g;
				done[ji_] = true;
			}
		}
	}
}


float SalMapMargolin::mean(std::vector<float>& v)
{
	const int n = (int)v.size();
	if (n == 0) return 0.f;

	float sum = (float)std::accumulate(std::begin(v), std::end(v), 0.0);
	return sum / n;
}


float SalMapMargolin::var(std::vector<float>& v)
{
	const int n = (int)v.size();
	if (n == 0) return 0.f;

	float _mean = mean(v);
	return std::accumulate(std::begin(v), std::end(v), 0.f,
		[&](const float b, const float e) {
		float diff = e - _mean;
		return b + diff * diff;
	}) / n;
}