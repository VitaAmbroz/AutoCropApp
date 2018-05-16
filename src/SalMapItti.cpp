/**
 * Bachelor thesis: Algorithms for automatic image cropping
 * VUT FIT 2018
 * Author: Vít Ambrož (xambro15@stud.fit.vutbr.cz)
 * Supervisor: Doc. Ing. Martin Čadík, Ph. D.
 * File: SalMapItti.cpp
 * Github repository: https://github.com/VitaAmbroz/AutoCropApp
 */

/*
 * This implementation of saliency map (Itti, L.; Koch, C.; Niebur, E.; A Model of Saliency-Based Visual Attention for Rapid Scene Analysis, 1998)
 * was inspired by https://github.com/akisato-/saliencyMap.
 */

#include "SalMapItti.h"

/**
 * Default constructor
 * @param src Original image
 */
SalMapItti::SalMapItti(cv::Mat src) {
	// set Gabor Kernel (9x9)
	this->GaborKernel0 = cv::Mat(9, 9, CV_32FC1, GaborKernel_0);
	this->GaborKernel45 = cv::Mat(9, 9, CV_32FC1, GaborKernel_45);
	this->GaborKernel90 = cv::Mat(9, 9, CV_32FC1, GaborKernel_90);
	this->GaborKernel135 = cv::Mat(9, 9, CV_32FC1, GaborKernel_135);
	
	// generate and save saliency map
	this->salMap = this->SMGetSM(src);
}


/**
 * Main method for generating saliency map.
 * @param src Original image
 * @return Matrix of output saliency map
 */
cv::Mat SalMapItti::SMGetSM(cv::Mat src) {
	int inputWidth = src.cols; // width of the image
	int inputHeight = src.rows; // height of the image
	cv::Size sSize = cv::Size(inputWidth, inputHeight);

	//=========================
	// Intensity and RGB Extraction
	//=========================
	this->SMExtractRGBI(src);

	//=========================
	// Feature Map Extraction (42 feature maps)
	//=========================
	// intensity feature maps(6)
	cv::Mat IFM[6];
	this->IFMGetFM(IFM);

	// color feature maps(12)
	cv::Mat CFM_RG[6];
	cv::Mat CFM_BY[6];
	this->CFMGetFM(CFM_RG, CFM_BY);

	// orientation feature maps(24)
	cv::Mat OFM[24];
	this->OFMGetFM(OFM);

	//=========================
	// Conspicuity Map Generation
	//=========================
	cv::Mat ICM = this->ICMGetCM(IFM, sSize);
	cv::Mat CCM = this->CCMGetCM(CFM_RG, CFM_BY, sSize);
	cv::Mat OCM = this->OCMGetCM(OFM, sSize);

	//=========================
	// Saliency Map Generation
	//=========================
	// Normalize conspicuity maps
	cv::Mat ICM_norm = this->SMNormalization(ICM);
	cv::Mat CCM_norm = this->SMNormalization(CCM);
	cv::Mat OCM_norm = this->SMNormalization(OCM);

	// Adding Intensity, Color, Orientation CM to form Saliency Map
	cv::Mat SM_Mat = cv::Mat(inputHeight, inputWidth, CV_32FC1); // Saliency Map matrix
	cv::addWeighted(ICM_norm, WEIGHT_INTENSITY, OCM_norm, WEIGHT_ORIENTATION, 0.0, SM_Mat);
	cv::addWeighted(CCM_norm, WEIGHT_COLOR, SM_Mat, 1.00, 0.0, SM_Mat);

	// Output Result Map
	cv::Mat normalizedSM = this->SMRangeNormalize(SM_Mat);
	cv::Mat smoothedSM = cv::Mat(SM_Mat.rows, SM_Mat.cols, CV_32FC1); // smoothed output
	cv::GaussianBlur(normalizedSM, smoothedSM, cv::Size(7, 7), 0, 0);

	cv::Mat SM = cv::Mat(inputHeight, inputWidth, CV_32FC1); // output is resized to original image
	cv::resize(smoothedSM, SM, SM.size(), 0, 0, CV_INTER_NN);
	
	// Normalize it to 0-255 values
	Mat SM_norm;
	normalize(SM, SM_norm, 0, 255, NORM_MINMAX, CV_8UC1);

	return SM_norm;
}


/**
 * Method for extracting color and intensity channels
 * @param inputImage Original image
 */
void SalMapItti::SMExtractRGBI(cv::Mat inputImage) {
	int height = inputImage.rows;
	int width = inputImage.cols;

	cv::Mat src(height, width, CV_32FC3);
	inputImage.convertTo(src, CV_32FC3, 1 / 256.0f);

	this->R = cv::Mat(height, width, CV_32FC1);
	this->G = cv::Mat(height, width, CV_32FC1);
	this->B = cv::Mat(height, width, CV_32FC1);
	this->I = cv::Mat(height, width, CV_32FC1);

	// split
	cv::Mat bgr[3];
	cv::split(src, bgr);
	this->B = bgr[0];
	this->G = bgr[1];
	this->R = bgr[2];

	// extract intensity image
	cv::cvtColor(src, this->I, CV_BGR2GRAY);
}

/**
 * Method for generating intensity feature maps
 * @param dst Destination pyramid of intensity feature maps
 */
void SalMapItti::IFMGetFM(cv::Mat dst[6]) {
	this->FMGaussianPyrCSD(this->I, dst);
}


/**
 * Method for generating color feature maps
 * @param RGFM Destination pyramid of red and green feature maps
 * @param RGFM Destination pyramid of blue and yellow feature maps
 */
void SalMapItti::CFMGetFM(cv::Mat RGFM[6], cv::Mat BYFM[6]) {
	int height = this->R.rows;
	int width = this->R.cols;

	cv::Mat tmp1 = cv::Mat(height, width, CV_32FC1);
	cv::Mat tmp2 = cv::Mat(height, width, CV_32FC1);
	cv::Mat RGBMax = cv::Mat(height, width, CV_32FC1);
	cv::Mat RGMin = cv::Mat(height, width, CV_32FC1);
	cv::Mat RGMat = cv::Mat(height, width, CV_32FC1);
	cv::Mat BYMat = cv::Mat(height, width, CV_32FC1);

	// Max(R,G,B)
	cv::max(this->R, this->G, tmp1);
	cv::max(this->B, tmp1, RGBMax);
	cv::max(RGBMax, 0.0001, RGBMax); // to prevent dividing by 0
	// Min(R,G)
	cv::min(this->R, this->G, RGMin);

	// tmp1 = R - G
	cv::subtract(this->R, this->G, tmp1);
	// tmp2 = B - Min(R,G)
	cv::subtract(this->B, RGMin, tmp2);
	// RG = (R-G) / Max(R,G,B)
	cv::divide(tmp1, RGBMax, RGMat);
	// BY = (B - Min(R,G) / Max(R,G,B)
	cv::divide(tmp2, RGBMax, BYMat);

	// Clamp negative value to 0 for the RG and BY maps
	cv::max(RGMat, 0, RGMat);
	cv::max(BYMat, 0, BYMat);

	// Obtain [RG,BY] color opponency feature map by generating Gaussian pyramid and performing center-surround difference
	this->FMGaussianPyrCSD(RGMat, RGFM);
	this->FMGaussianPyrCSD(BYMat, BYFM);
}


/**
 * Method for generating orientation feature maps
 * @param dst Destination pyramid of orientation feature maps
 */
void SalMapItti::OFMGetFM(cv::Mat dst[24]) {
	// Create gaussian pyramid
	cv::Mat GaussianI[9];
	this->FMCreateGaussianPyr(I, GaussianI);

	// Convolution Gabor filter with intensity feature maps to extract orientation feature
	cv::Mat tempGaborOutput0[9];
	cv::Mat tempGaborOutput45[9];
	cv::Mat tempGaborOutput90[9];
	cv::Mat tempGaborOutput135[9];
	for (int j = 2; j < 9; j++) {
		int now_height = GaussianI[j].rows;
		int now_width = GaussianI[j].cols;
		tempGaborOutput0[j] = cv::Mat(now_height, now_width, CV_32FC1);
		tempGaborOutput45[j] = cv::Mat(now_height, now_width, CV_32FC1);
		tempGaborOutput90[j] = cv::Mat(now_height, now_width, CV_32FC1);
		tempGaborOutput135[j] = cv::Mat(now_height, now_width, CV_32FC1);
		cv::filter2D(GaussianI[j], tempGaborOutput0[j], GaussianI[j].depth(), this->GaborKernel0);
		cv::filter2D(GaussianI[j], tempGaborOutput45[j], GaussianI[j].depth(), this->GaborKernel45);
		cv::filter2D(GaussianI[j], tempGaborOutput90[j], GaussianI[j].depth(), this->GaborKernel90);
		cv::filter2D(GaussianI[j], tempGaborOutput135[j], GaussianI[j].depth(), this->GaborKernel135);
	}

	// calculate center surround difference for each orientation
	cv::Mat temp0[6];
	cv::Mat temp45[6];
	cv::Mat temp90[6];
	cv::Mat temp135[6];
	this->FMCenterSurroundDiff(tempGaborOutput0, temp0);
	this->FMCenterSurroundDiff(tempGaborOutput45, temp45);
	this->FMCenterSurroundDiff(tempGaborOutput90, temp90);
	this->FMCenterSurroundDiff(tempGaborOutput135, temp135);

	// saving the 6 center-surround difference feature map of each angle configuration to the destination pointer
	for (int i = 0; i < 6; i++) {
		dst[i] = temp0[i];
		dst[i + 6] = temp45[i];
		dst[i + 12] = temp90[i];
		dst[i + 18] = temp135[i];
	}
}


/**
 * Method for generating intensity conspicuity map
 * @param IFM Destination pyramid of intensity conspicuity maps
 * @param size Size of original image
 * @return Intensity conspicuity map
 */
cv::Mat SalMapItti::ICMGetCM(cv::Mat IFM[6], cv::Size size) {
	const int num_FMs = 6;

	// Normalize all intensity feature maps
	cv::Mat NIFM[num_FMs];
	this->normalizeFeatureMaps(IFM, NIFM, size.width, size.height, num_FMs);

	// Formulate intensity conspicuity map by summing up the normalized intensity feature maps
	cv::Mat ICM = cv::Mat(size.height, size.width, CV_32FC1, float(0)); // init with zeros
	for (int i = 0; i < num_FMs; i++) {
		//NIFM[i].convertTo(NIFM[i], CV_32FC1); there used to be some problem with types and sizes, this line used to fix it
		cv::add(ICM, NIFM[i], ICM);
	}

	return ICM;
}


/**
 * Method for generating color conspicuity map
 * @param CFM_RG Source pyramid of red and green feature maps
 * @param CFM_BY Source pyramid of blue and yellow feature maps
 * @param size Size of original image
 * @return Color conspicuity map
 */
cv::Mat SalMapItti::CCMGetCM(cv::Mat CFM_RG[6], cv::Mat CFM_BY[6], cv::Size size) {
	cv::Mat CCM_RG = ICMGetCM(CFM_RG, size);
	cv::Mat CCM_BY = ICMGetCM(CFM_BY, size);

	cv::Mat CCM = cv::Mat(size.height, size.width, CV_32FC1);
	cv::add(CCM_BY, CCM_RG, CCM);

	return CCM;
}


/**
 * Method for generating orientation conspicuity map
 * @param OFM Source pyramid of orientation feature maps
 * @param size Size of original image
 * @return Orientation conspicuity map
 */
cv::Mat SalMapItti::OCMGetCM(cv::Mat OFM[24], cv::Size size) {
	int num_FMs_perAngle = 6;
	int num_angles = 4;
	int num_FMs = num_FMs_perAngle * num_angles;

	// split feature maps into four sets
	cv::Mat OFM0[6];
	cv::Mat OFM45[6];
	cv::Mat OFM90[6];
	cv::Mat OFM135[6];
	for (int i = 0; i < num_FMs_perAngle; i++) {
		OFM0[i] = OFM[0 * num_FMs_perAngle + i];
		OFM45[i] = OFM[1 * num_FMs_perAngle + i];
		OFM90[i] = OFM[2 * num_FMs_perAngle + i];
		OFM135[i] = OFM[3 * num_FMs_perAngle + i];
	}

	// extract conspicuity map for each angle
	cv::Mat NOFM_tmp[4];
	NOFM_tmp[0] = ICMGetCM(OFM0, size);
	NOFM_tmp[1] = ICMGetCM(OFM45, size);
	NOFM_tmp[2] = ICMGetCM(OFM90, size);
	NOFM_tmp[3] = ICMGetCM(OFM135, size);

	// Normalize all orientation features map grouped by their orientation angles
	cv::Mat NOFM[4];
	for (int i = 0; i<4; i++) {
		NOFM[i] = SMNormalization(NOFM_tmp[i]);
	}

	// Sum up all orientation feature maps, and form orientation conspicuity map
	cv::Mat OCM = cv::Mat(size.height, size.width, CV_32FC1, float(0));

	for (int i = 0; i<4; i++) {
		//NOFM[i].convertTo(NOFM[i], CV_32FC1); there used to be some problem with sizes and types, this line used to fix it
		cv::add(NOFM[i], OCM, OCM);
	}

	return OCM;
}


void SalMapItti::FMGaussianPyrCSD(cv::Mat src, cv::Mat dst[6]) {
	cv::Mat GaussianMap[9];
	this->FMCreateGaussianPyr(src, GaussianMap);
	this->FMCenterSurroundDiff(GaussianMap, dst);
}


void SalMapItti::FMCreateGaussianPyr(cv::Mat src, cv::Mat dst[9]) {
	dst[0] = src.clone();

	for (int i = 1; i < 9; i++) {
		dst[i] = cv::Mat(dst[i - 1].rows / 2, dst[i - 1].cols / 2, CV_32FC1);
		cv::pyrDown(dst[i - 1], dst[i]); // blurs an image and downsamples it.
	}
}


void SalMapItti::FMCenterSurroundDiff(cv::Mat GaussianMap[9], cv::Mat dst[6]) {
	int i = 0;
	for (int s = 2; s < 5; s++) {
		int now_height = GaussianMap[s].rows;
		int now_width = GaussianMap[s].cols;

		cv::Mat tmp = cv::Mat(now_height, now_width, CV_32FC1);

		dst[i] = cv::Mat(now_height, now_width, CV_32FC1);
		dst[i + 1] = cv::Mat(now_height, now_width, CV_32FC1);

		cv::resize(GaussianMap[s + 3], tmp, tmp.size(), 0, 0, CV_INTER_LINEAR);
		cv::absdiff(GaussianMap[s], tmp, dst[i]);
		cv::resize(GaussianMap[s + 4], tmp, tmp.size(), 0, 0, CV_INTER_LINEAR);
		cv::absdiff(GaussianMap[s], tmp, dst[i + 1]);

		i += 2;
	}
}


void SalMapItti::normalizeFeatureMaps(cv::Mat FM[6], cv::Mat NFM[6], int width, int height, int num_maps) {
	
	for (int i = 0; i<num_maps; i++) {
		cv::Mat normalizedImage = SMNormalization(FM[i]);

		NFM[i] = cv::Mat(height, width, CV_32FC1);
		cv::resize(normalizedImage, NFM[i], NFM[i].size(), 0, 0, CV_INTER_LINEAR);
	}
}


cv::Mat SalMapItti::SMNormalization(cv::Mat src) {
	cv::Mat result = cv::Mat(src.rows, src.cols, CV_32FC1);

	// normalize so that the pixel value lies between 0 and 1
	cv::Mat tempResult = this->SMRangeNormalize(src);

	// single-peak emphasis / multi-peak suppression
	double lmaxmean = this->SMAvgLocalMax(tempResult);
	double normCoeff = (1 - lmaxmean)*(1 - lmaxmean);

	tempResult.convertTo(result, CV_32FC1, normCoeff);

	return result;
}


cv::Mat SalMapItti::SMRangeNormalize(cv::Mat src) {
	double maxx, minn;
	cv::minMaxLoc(src, &minn, &maxx);

	cv::Mat result = cv::Mat(src.rows, src.cols, CV_32FC1);
	if (maxx != minn)
		src.convertTo(result, CV_32FC1, 1 / (maxx - minn), minn / (minn - maxx));
	else
		src.convertTo(result, CV_32FC1, 1, -minn);

	return result;
}


double SalMapItti::SMAvgLocalMax(cv::Mat src) {
	CvMat cvSrc = src; // overtyping to C format CvMat

	int stepsize = DEFAULT_STEP_LOCAL;
	int numlocal = 0;
	double lmaxmean = 0, lmax = 0, dummy = 0;
	CvMat localMatHeader;
	cvInitMatHeader(&localMatHeader, stepsize, stepsize, CV_32FC1, cvSrc.data.ptr, cvSrc.step);

	for (int y = 0; y < src.rows - stepsize; y += stepsize) { // Note: the last several pixels may be ignored.
		for (int x = 0; x < src.cols - stepsize; x += stepsize) {
			localMatHeader.data.ptr = cvSrc.data.ptr + sizeof(float) * x + cvSrc.step * y; // get local matrix by pointer trick
			cvMinMaxLoc(&localMatHeader, &dummy, &lmax);
			lmaxmean += lmax;
			numlocal++;
		}
	}

	return lmaxmean / numlocal;
}