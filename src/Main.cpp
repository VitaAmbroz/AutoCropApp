/**
 * Bachelor thesis: Algorithms for automatic image cropping
 * VUT FIT 2018
 * Author: Vít Ambrož (xambro15@stud.fit.vutbr.cz)
 * Supervisor: Doc. Ing. Martin Čadík, Ph. D.
 * File: Main.cpp
 * Github repository: https://github.com/VitaAmbroz/AutoCropApp
 */

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "Arguments.h"
#include "SalMapStentiford.h"
#include "SalMapMargolin.h"
#include "SalMapItti.h"
#include "CompositionModel.h"
#include "AutocropStentiford.h"
#include "AutocropFang.h"
#include "AutocropSuh.h"

using namespace std;
using namespace cv;

/* prototypes of functions */
void showImageAuto(std::string title, const Mat& img);

/* constant for help message */
const char* HELP_MESSAGE = 
"-----------------------------------------------------------------------------------\n"
"This is application for automatic image cropping.\n"
"See README or xambro15.pdf for detail description of implemented algorithms.\n\n"
"Possible ways to run this application:\n"
" $ ./autocrop -help  => Displays this message.\n"
" $ ./autocrop imagePath  => Crops image with all algorithms(imagePath is path to source image).\n"
" $ ./autocrop imagePath -suh  => Uses Suh's algorithm of automatic image cropping.\n"
" $ ./autocrop imagePath -sten  => Uses Stentiford's algorithm of automatic image cropping.\n"
" $ ./autocrop imagePath -fang  => Uses Fang's algorithm of automatic image cropping.\n"
" $ ./autocrop imagePath -wh 600 400  => Defines width and height of final crop(in pixels).\n"
" $ ./autocrop imagePath -scale 0.5  => Scales down original image to final crop(keeps aspect ratio).\n"
" $ ./autocrop imagePath -whratio 16 9  => Defines aspect ratio of final crop(width:height).\n"
" $ ./autocrop imagePath -suh -threshold 0.5  => Defines threshold used in Suh's algorithm.\n"
" $ ./autocrop imagePath -w  => Disables showing windows of saliency maps and gradient.\n"
" $ ./autocrop -train datasetPath  => Runs training of Visual Composition model. datasetPath is path to directory with images.\n\n"
"xambro15@stud.fit.vutbr.cz, VUT FIT 2018\n"
"-----------------------------------------------------------------------------------";

int main(int argc, char** argv)
{
	// parse command line arguments and save flags
	Arguments arguments(argc, argv);
	// check if help message should be displayed
	if (arguments.isHelpActivated()) {
		std::cout << std::string(HELP_MESSAGE) << std::endl;
		std::exit(EXIT_SUCCESS);
	}
	// check if parsing of arguments was correct 
	if (!arguments.isAllClear()) {
		std::cerr << "Run program with argument -help to see list of possible arguments" << std::endl;
		std::exit(EXIT_FAILURE);
	}
	// run training pipeline for creating model of Visual composition
	if (arguments.runTraining) {
		CompositionModel comp;
		comp.fullTrainingPipeline(arguments.trainingDatasetPath, "./models/Trained_model.yml");
		std::exit(EXIT_SUCCESS);
	}


	// load original Image
	cv::Mat img = cv::imread(arguments.imgPath, CV_LOAD_IMAGE_COLOR);
	if (img.empty()) {
		std::cerr << "Function imread() failed to open target image!" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	// default values for construction of grid for top left corner positions
	int HSTEP = img.cols / 100;
	int VSTEP = img.rows / 100;
	if (HSTEP < 1) HSTEP = 1;
	if (VSTEP < 1) VSTEP = 1;

	// show original image
	showImageAuto("original", img);

	// if no method was specified in argument, make all of them one by one
	if (!arguments.isSuh() && !arguments.isStentiford() && !arguments.isFang()) {
		arguments.setSuh();
		arguments.setStentiford();
		arguments.setFang();
	}

	/***************************************************/
	/* Automatic thumbnail cropping and its effectiveness(Suh et al.; 2003) */
	if (arguments.isSuh()) {
		// generate Itti's saliency map(1998) and display it
		std::cout << "\nGenerating saliency map by Itti..." << std::endl;
		SalMapItti itti(img);
		if (arguments.isWindowsEnabled()) {
			showImageAuto("IttiSM", itti.salMap);
		}

		// automatic cropping methods
		std::cout << "\nLooking for the best cropping window..." << std::endl;
		AutocropSuh suh(itti.salMap);

		// use default threshold or threshold specified in arguments
		const float THRESHOLD = 0.6f;
		float tr = (arguments.isThreshold()) ? arguments.getThreshold() : THRESHOLD;

		if (arguments.isWH())
			suh.bruteForceWH(arguments.getWidth(), arguments.getHeight(), HSTEP, VSTEP);
		else if (arguments.isScale())
			suh.bruteForceScale(arguments.getScale(), HSTEP, VSTEP);
		else if (arguments.isWHratio())
			suh.bruteForceWHratio(arguments.getWidthRatio(), arguments.getHeightRatio(), tr, HSTEP, VSTEP);
		else
			suh.greedyGeneral(tr);
			//suh.bruteForceGeneral(tr);

		// define region of interest for cropping
		cv::Rect roi(suh.getX(), suh.getY(), suh.getWidth(), suh.getHeight());
		// crop the original image to the defined roi
		cv::Mat cropSuh = img(roi);
		// show cropped result in window
		showImageAuto("SuhAutocrop", cropSuh);
		// save cropped image
		cv::imwrite("cropSuh.jpg", cropSuh);
	}

	/***************************************************/
	/* Attention based auto image cropping(Stentiford, F.; 2007) */
	if (arguments.isStentiford()) {
		// generate saliency map(Stentiford, F.: Attention-based auto image cropping, 2007)
		SalMapStentiford StentifordSM(img);
		std::cout << "\nGenerating saliency map by Stentiford..." << std::endl;
		StentifordSM.generateSalMap();
		
		if (arguments.isWindowsEnabled()) {
			showImageAuto("StentifordSalMap", StentifordSM.salMap);
		}

		// automatic cropping methods
		std::cout << "\nLooking for the best cropping window..." << std::endl;
		AutocropStentiford abStentiford(StentifordSM.salMap);

		// value used to define minimum size for ROI in methods with random generator
		const float DEFAULT_ZOOM_FACTOR = 1.5f; 

		if (arguments.isWH())
			abStentiford.brutalForceWH(arguments.getWidth(), arguments.getHeight(), HSTEP, VSTEP);
		else if (arguments.isScale())
			abStentiford.brutalForceZoomFactor((1.0f / arguments.getScale()), HSTEP, VSTEP);
		else if (arguments.isWHratio())
			abStentiford.randomWHratio(arguments.getWidthRatio(), arguments.getHeightRatio(), DEFAULT_ZOOM_FACTOR);
		else 
			abStentiford.randomZFWalk(DEFAULT_ZOOM_FACTOR); // using fast method with defined number of generated ROIs
		//abStentiford.zoomFactorWalk(1.5, 2.0, 0.1, HSTEP, VSTEP);
		//abStentiford.randomWalk(600, 600);
		
		// define region of interest for cropping
		cv::Rect roi(abStentiford.getX(), abStentiford.getY(), abStentiford.getWidth(), abStentiford.getHeight());
		// crop the original image to the defined roi
		cv::Mat cropStentiford = img(roi);
		// show cropped result in window
		showImageAuto("StentifordAutocrop", cropStentiford);
		// save cropped image
		cv::imwrite("cropStentiford.jpg", cropStentiford);
	}

	/***************************************************/
	/* Automatic Image Cropping using Visual Composition, Boundary Simplicity and Content Preservation Models (Fang et al.; 2014)*/
	if (arguments.isFang()) {
		// generate saliency map(Margolin, R.; Tal, A.; Zelnik-Manor, L.: What Makes a Patch Distinct?, 2013)
		std::cout << "\nGenerating saliency map by Margolin..." << std::endl;
		SalMapMargolin MargolinSM(img);
		
		if (arguments.isWindowsEnabled()) {	// show saliency map
			showImageAuto("MargolinSalMap", MargolinSM.salMap);
		}

		// automatic cropping methods
		AutocropFang fang(img, MargolinSM.salMap, "./models/Trained_model21.yml");
		
		if (arguments.isWindowsEnabled()) {	// show gradient map
			showImageAuto("ImageGradient", fang.gradient);
		}
		
		std::cout << "\nLooking for the best cropping window..." << std::endl;
		if (arguments.isWH())
			fang.WHCrop(arguments.getWidth(), arguments.getHeight(), HSTEP, VSTEP);
		else if (arguments.isScale())
			fang.scaleCrop(arguments.getScale(), HSTEP, VSTEP);
		else if (arguments.isWHratio())
			fang.WHratioCrop(arguments.getWidthRatio(), arguments.getHeightRatio(), HSTEP, VSTEP);
		else
			fang.randomGridCrop();

		// Define region of interest for cropping
		cv::Rect roi2(fang.getX(), fang.getY(), fang.getWidth(), fang.getHeight());
		// Crop the original image to the defined roi
		cv::Mat cropFang = img(roi2);
		// Show cropped result in window
		showImageAuto("FangAutocrop", cropFang);
		// save cropped image
		cv::imwrite("cropFang.jpg", cropFang);
	}

	return 0;
}


/**
 * Function for showing window with image in normalized size
 * @param title Name of the window
 * @param img Image to be displayed
 */
void showImageAuto(std::string title, const Mat &img)
{
	std::cout << "Showing image: \"" << title << "\". Press any key to continue..." << std::endl;
	cv::namedWindow(title, CV_WINDOW_AUTOSIZE);

	double scale = 800.f / max(img.cols, img.rows);

	if (scale >= 1.f)	// dont resize
		cv::imshow(title, img);
	else {
		// image is large, resize it to max 800px of width or height
		cv::Mat scaled;
		// use INTER_AREA to resampling using pixel area relation
		cv::resize(img, scaled, Size(), scale, scale, cv::INTER_AREA);
		cv::imshow(title, scaled);
	}
	cv::waitKey(0);
}