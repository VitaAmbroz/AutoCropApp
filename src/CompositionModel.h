/**
 * Bachelor thesis: Algorithms for automatic image cropping
 * VUT FIT 2018
 * Author: Vít Ambrož (xambro15@stud.fit.vutbr.cz)
 * Supervisor: Doc. Ing. Martin Čadík, Ph. D.
 * File: CompositionModel.h
 * Github repository: https://github.com/VitaAmbroz/AutoCropApp
 */
#ifndef __COMPOSITION_H__
#define __COMPOSITION_H__

#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include "boost/filesystem.hpp"

#include "SalMapMargolin.h"

namespace fs = boost::filesystem;
using namespace std;
using namespace cv;

/* classes constants for good and bad crops */
const int GOOD_CROP = 1;
const int BAD_CROP = 0;

/* count of features used for training and classifying in Visual composition model */
//const int FEATS_COUNT = 85; // SPSM 8x8 + 4x4 + 2x2 + 1x1
const int FEATS_COUNT = 21; // SPSM 4x4 + 2x2 + 1x1


class CompositionModel
{
public:
	// constructor
	CompositionModel();

	// method for training SVM model
	void fullTrainingPipeline(fs::path srcDir, std::string trainedSavepath);
	// methods for creating and managing feature vectors
	void createFeatureMat(fs::path srcDir); 
	void loadFeatureMat(std::string filepath);
	void addFeatureVector(cv::Mat saliency, int cls);
	cv::Mat getFeatureVector(cv::Mat saliency);
	// method for training and mananing SVM model
	void train(std::string savepath);
	void loadTrainedModel(std::string filepath);
	float classifyComposition(cv::Mat featVec);

private:
	// matrix/vectors with features and classes
	cv::Mat featMat;
	cv::Mat classMat;
	// SVM model
	cv::Ptr<cv::ml::SVM> model;
	
	// other methods that are used for operations with images in training pipeline
	cv::Rect randomCrop(cv::Mat img);
	cv::Mat loadImgReduced(std::string path);
	std::vector<fs::path> getImagePaths(fs::path dir);
};

#endif //__COMPOSITION_H__