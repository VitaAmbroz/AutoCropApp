/**
 * Bachelor thesis: Algorithms for automatic image cropping
 * VUT FIT 2018
 * Author: Vít Ambrož (xambro15@stud.fit.vutbr.cz)
 * Supervisor: Doc. Ing. Martin Čadík, Ph. D.
 * File: CompositionModel.cpp
 * Github repository: https://github.com/VitaAmbroz/AutoCropApp
 */

/**
* This is implementation of methods that were necessary for creating Visual Composition model.
* I write most parts of this implementation, but some key parts(using cv::ml::SVM class and creating feature matrix)
* were inspired by https://github.com/swook/autocrop/tree/master/src
*/

#include "CompositionModel.h"

/**
* Constructor for Visual Composition model(Fang et al. 2014)
*/
CompositionModel::CompositionModel() {
	this->featMat = Mat(Size(FEATS_COUNT, 0), CV_32FC1);
	this->classMat = Mat(Size(1, 0), CV_32FC1);
}


/**
* Creates feature matrix of dataset images and then runs SVM training
* @param srcDir Path to dataset folder with well composed images.
* @param trainedSavepath Path where trained model will be saved.
*/
void CompositionModel::fullTrainingPipeline(fs::path srcDir, std::string trainedSavepath) {
	this->createFeatureMat(srcDir);
	this->train(trainedSavepath);
}


/**
* Creates feature matrix from feature vectors of well and also bad composed images.
* @param srcDir Path to dataset folder with well composed images.
*/
void CompositionModel::createFeatureMat(fs::path srcDir) {
	srand((unsigned int)time(NULL));
	// save paths for all images in training dataset of well composed images
	std::vector<fs::path> paths = this->getImagePaths(srcDir);

#pragma omp parallel for
	for (int i = 0; i < paths.size(); i++) {
		std::cout << i << ") Processing " << paths.at(i).string() << std::endl;	// full path
		//std::cout << "filename and extension: " << paths.at(i).filename() << std::endl; // img.jpg
		//std::cout << "filename only: " << paths.at(i).stem() << std::endl;     // img

		Mat img = this->loadImgReduced(paths.at(i).string());
		if (!img.data) {
			std::cerr << "Error reading: " << paths[i] << std::endl;
			continue;
		}

		// generate saliency map of well composed image
		SalMapMargolin saliency(img);
		// add well composed image as feature vector
		cv::Mat fVecGood = this->getFeatureVector(saliency.salMap);
		this->addFeatureVector(fVecGood, GOOD_CROP);

		// randomly generated crop should be bad composed => define area and content conditions
		float sumSaliency = (float)sum(saliency.salMap)[0];
		float totalArea = (float)(img.cols * img.rows);
		cv::Rect rect;
		cv::Mat badCrop;

		while (true) {
			rect = this->randomCrop(saliency.salMap); // create random ROI for crop
			badCrop = saliency.salMap(rect); // crop defined ROI

			// saliency ratio = bad cropped image : original image
			float S_content = (float)(sum(badCrop)[0] / sumSaliency);
			// area ratio = area of cropped ROI : area of original image
			float S_area = (float)(rect.area() / totalArea);

			if (S_content < 0.25 && S_area > 0.25)  // conditions mean that the crop will be probably bad
			{
				// add bad composed crop as feature vector
				cv::Mat fVecBad = this->getFeatureVector(badCrop);
				this->addFeatureVector(fVecBad, BAD_CROP);
				// possibly save bad crop
				/* std::string path = srcDir.string() + "/badCrops21/" + paths.at(i).stem().string() + ".jpg";
				std::cout << "Saving bad composed image " << path << std::endl;
				cv::imwrite(path, img(rect)); */
				break;
			}
		}
	}

	// save feature matrix to file storage(format .yml)
	FileStorage fs = FileStorage(srcDir.string() + "/FeatMat.yml", FileStorage::WRITE | FileStorage::FORMAT_YAML);
	cv::write(fs, "features", this->featMat);
	cv::write(fs, "classes", this->classMat);
}


/**
* Loads data that represents feature matrix of dataset images.
* @param filepath Path for .yml file that contains features of dataset images
*/
void CompositionModel::loadFeatureMat(std::string filepath) {
	std::cout << "Loading feature matrix: " << filepath << std::endl;
	
	cv::FileStorage fs = FileStorage(filepath, FileStorage::READ | FileStorage::FORMAT_YAML);
	if (!fs.isOpened()) {
		std::cerr << "Cannot load file " << filepath << " as feature matrix." << std::endl;
		std::exit(EXIT_FAILURE);
	}

	// store .yml data
	fs["features"] >> this->featMat;
	fs["classes"] >> this->classMat;

	// conditions for loaded data
	if ((this->featMat.cols != FEATS_COUNT) || (this->featMat.rows != this->classMat.rows) || (this->classMat.cols != 1)) {
		std::cerr << "Bad format of data in loaded feature matrix." << std::endl;
		std::exit(EXIT_FAILURE);
	}
}


/**
* Adds SPSM(Spatial Pyramid of Saliency Map) feature vector into feature matrix.
* @param fVec Feature vector to be added into feature matrix.
* @param cls Class of feature vector - it could be only GOOD_CROP or BAD_CROP
*/
void CompositionModel::addFeatureVector(cv::Mat fVec, int cls) {
	if (featMat.cols != FEATS_COUNT) return;
	if (cls != GOOD_CROP && cls != BAD_CROP) return;

	this->featMat.push_back(fVec);
	this->classMat.push_back(cls);
}


/**
* Create and returns feature vector as Spatial Pyramid of input saliency map
* @param saliency Saliency map from that will be created feature vector
* @return Feature vector (Matrix with FEATS_COUNT cols and 1 row)
*/
cv::Mat CompositionModel::getFeatureVector(cv::Mat saliency) {
	cv::Mat _saliency;
	
	if (FEATS_COUNT == 85)  // Resize saliency map to be 8x8. Use INTER_AREA to average pixel values
		resize(saliency, _saliency, cv::Size(8, 8), cv::INTER_AREA);
	else // Resize saliency map to be 4x4. Use INTER_AREA to average pixel values
		resize(saliency, _saliency, cv::Size(4, 4), cv::INTER_AREA);

	// pointer to first pixel data
	float* p_saliency = _saliency.ptr<float>(0);

	/*namedWindow("window", WINDOW_NORMAL);
	resizeWindow("window", 400, 400);
	imshow("window", _saliency);*/

	// Initialise feature vector
	cv::Mat feats = cv::Mat(cv::Size(FEATS_COUNT, 1), CV_32FC1);
	// pointer to first pixel data
	float* p_feats = feats.ptr<float>(0);

	// Index in feature vector
	int i = 0;
	
	if (FEATS_COUNT == 85) {
		// Store 1/64ths as feature vector values
		for (int j = 0; j < 64; j++) {
			p_feats[i] = p_saliency[i];
			i++;
		}

		// Average 1/64 values to get feature vector values for 1/16ths
		for (int j = 0; j < 4; j++) {
			for (int k = 0; k < 4; k++) {
				p_feats[i] = 0.25f * (p_feats[16 * j + 2 * k] +
					p_feats[16 * j + 2 * k + 1] +
					p_feats[16 * j + 2 * k + 8] +
					p_feats[16 * j + 2 * k + 9]);
				i++;
			}
		}

		// Average 1/16 values to get feature vector values for 1/4ths
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {
				p_feats[i] = 0.25f * (p_feats[64 + 8 * j + 2 * k] +
									p_feats[64 + 8 * j + 2 * k + 1] +
									p_feats[64 + 8 * j + 2 * k + 4] +
									p_feats[64 + 8 * j + 2 * k + 5]);
				i++;
			}
		}
	}
	else {
		// Store 1/16ths as feature vector values
		for (int j = 0; j < 16; j++) {
			p_feats[i] = p_saliency[i];
			i++;
		}

		// Average 1/16 values to get feature vector values for 1/4ths
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {
				p_feats[i] = 0.25f * (p_feats[8 * j + 2 * k] +
									p_feats[8 * j + 2 * k + 1] +
									p_feats[8 * j + 2 * k + 4] +
									p_feats[8 * j + 2 * k + 5]);
				i++;
			}
		}
	}

	// Average 1/4 value to get feature value for whole image
	p_feats[i] = 0.25f * (p_feats[i - 4] +
						p_feats[i - 3] +
						p_feats[i - 2] +
						p_feats[i - 1]);

	return feats;
}


/**
* Method for training model using opencv ml::SVM class.
* @param savepath Path where trained model will be saved.
*/
void CompositionModel::train(std::string savepath) {
	// create instance of cv::ml::SVM
	this->model = ml::SVM::create();

	// add data from feature matrix and class vector
	auto traindata = ml::TrainData::create(this->featMat, ml::ROW_SAMPLE, this->classMat);
	std::cout << "> Training with " << this->featMat.rows << " rows of data." << std::endl;
	std::cout << "> Training with " << this->featMat.cols << " features." << std::endl;

	// this part was inspired by https://github.com/swook/autocrop/blob/master/src/training/Trainer.cpp
	// Set default SVM parameters
	model->setType(ml::SVM::C_SVC);
	model->setKernel(ml::SVM::LINEAR);
	// Parameter C of a SVM optimization problem
	model->setC(10.f);

	// Set candidate hyperparameter values
	ml::ParamGrid CGrid = ml::SVM::getDefaultGrid(ml::SVM::C),
				gammaGrid = ml::SVM::getDefaultGrid(ml::SVM::GAMMA),
				pGrid = ml::SVM::getDefaultGrid(ml::SVM::P),
				nuGrid = ml::SVM::getDefaultGrid(ml::SVM::NU),
				coeffGrid = ml::SVM::getDefaultGrid(ml::SVM::COEF),
				degreeGrid = ml::SVM::getDefaultGrid(ml::SVM::DEGREE);

	int  kFold = 25;   // K-fold Cross-Validation
	bool balanced = true;  // the method creates more balanced cross-validation subsets
	CGrid.logStep = 1.05f;  // Logarithmic step for iterating the statmodel parameter
	CGrid.minVal = 1e-4;  // Maximum value of the statmodel parameter.
	CGrid.maxVal = 1e+4;  // Minimum value of the statmodel parameter

	// Run cross-validation with SVM
	model->trainAuto(traindata, kFold, CGrid, gammaGrid, pGrid, nuGrid, coeffGrid, degreeGrid, balanced);

	// Calculate error on whole dataset
	float err; 
	err = model->calcError(traindata, false, noArray());
	std::cout << "> Training error on whole dataset: " << err << std::endl;

	// save trained model
	this->model->save(savepath);
}


/**
* Loads trained model
* @param filepath Path to file with trained model data (serialized SVM)
*/
void CompositionModel::loadTrainedModel(std::string filepath) {
	std::cout << "Loading trained SVM model: " << filepath << std::endl;
	try {
		this->model = ml::SVM::load(filepath);
	}
	catch(std::exception e) {
		std::cerr << "Cannot load trained model: " << filepath << std::endl;
		std::exit(EXIT_FAILURE);
	}
}


/**
* Classifies input feature vector using SVR(Support vector regression)
* @param featVec Feature vector to be classified according to trained model data
* @return Composition score in interval (-1,+1), where -1 means the best composition and +1 the worst composition
*/
float CompositionModel::classifyComposition(cv::Mat featVec)
{
	// use flag StatModel::RAW_OUTPUT to get the raw response from SVM (in the case of regression)
	cv::Mat result;
	this->model->predict(featVec, result, ml::StatModel::RAW_OUTPUT);

	return result.at<float>(0, 0);
}


/**
* Method for finding ROI to crop randomly -> it should represent bad composed image 
* @param img Image to be cropped
* @return Rectangle that represents ROI to crop
*/
cv::Rect CompositionModel::randomCrop(cv::Mat img) {
	// define minimum size
	int minW = (int)(0.5 * img.cols);
	int minH = (int)(0.5 * img.rows);

	// generate random ROI
	int x1, y1, w, h;
	x1 = rand() % (img.cols - minW);
	y1 = rand() % (img.rows - minH);
	w = minW + (rand() % (img.cols - x1 - minW));
	h = minH + (rand() % (img.rows - y1 - minH));

	return cv::Rect(x1, y1, w, h);
}


/**
* Method for scaling image if it is too large
* @param path Path to image
* @return Normalized image that is not larger than 1000px
*/
cv::Mat CompositionModel::loadImgReduced(std::string path)
{
	cv::Mat img = cv::imread(path, CV_LOAD_IMAGE_COLOR);

	double scale = 1000.f / max(img.cols, img.rows);
	// if is smaller than defined limit, return original
	if (scale >= 1.f) return img;

	// too large -> resize it using INTER_AREA to resampling using pixel area relation
	cv::resize(img, img, Size(), scale, scale, cv::INTER_AREA);
	return img;
}


/**
* Saves paths of all files in selected directory to vector
* @param dir Path to directory (in this case it would be image dataset directory)
* @return Vector containing paths to images in selected directory
*/
std::vector<fs::path> CompositionModel::getImagePaths(fs::path dir) {
	if (!is_directory(dir) || !fs::exists(dir)) {
		std::cerr << "Invalid path of dataset directory!" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	
	std::vector<fs::path> out;
	fs::directory_iterator end;
	for (fs::directory_iterator it(dir); it != end; it++)
	{
		if (fs::is_regular_file(it->status()))
		{
			out.push_back(*it);
		}
	}

	return out;
}