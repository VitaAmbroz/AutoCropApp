/**
 * Bachelor thesis: Algorithms for automatic image cropping
 * VUT FIT 2018
 * Author: Vít Ambrož (xambro15@stud.fit.vutbr.cz)
 * Supervisor: Doc. Ing. Martin Čadík, Ph. D.
 * File: Arguments.cpp
 * Github repository: https://github.com/VitaAmbroz/AutoCropApp
 */

#include "Arguments.h"

/**
 * Constructor for parsing arguments and initaliazing default values.
 */
Arguments::Arguments(int mArgc, char** mArgv) {
    this->help = false;
    this->allClear = true;
    this->enableWindows = true;

    this->runTraining = false;
    this->trainingDatasetPath = "";

    this->suh = false;
    this->stentiford = false;
    this->fang = false;
    this->wh = false;
    this->scale = false;
    this->whRatio = false;
    this->threshold = false;

    this->width = 0;
    this->height = 0;
    this->scaleValue = 0.0f;
    this->wRatio = 0;
    this->hRatio = 0;
    this->suhThreshold = 0.0f;

    this->imgPath = "";
    this->argc = mArgc;
    this->argv = mArgv;
    this->parse();
}


/**
 * Function for parsing arguments and saving selected flags
 */ 
void Arguments::parse() {
    // check if there is not too few arguments
    if (this->argc < 2) {
        this->allClear = false;
        std::cerr << "Too few arguments! Path of input image is missing!" << std::endl;
        return;
    }

    // only 2 arguments <=> help or input image only
    if (this->argc == 2) {
        // check if help is requested
        if (std::string(this->argv[1]) == "-h" || std::string(this->argv[1]) == "-help" || std::string(this->argv[1]) == "--help") {
            this->help = true;
        }
        else if (std::string(this->argv[1]) == "-train" || std::string(this->argv[1]) == "train" || std::string(this->argv[1]) == "-training") {
            std::cerr << "Undefined path of training dataset!" << std::endl;
            this->allClear = false;
        }
        else { // the first argument must be path of input image
            this->imgPath = this->argv[1];
        }
    }
    else { // more arguments
        // check if training was specified $ ./autocrop -train path
        if (std::string(this->argv[1]) == "-train" || std::string(this->argv[1]) == "train" || std::string(this->argv[1]) == "-training") {
            this->runTraining = true;
            this->trainingDatasetPath = this->argv[2];

            return;
        }


        this->imgPath = this->argv[1]; // first argument must be path of input image

        // loop in list of arguments
        for (int i = 2; i < this->argc; i++) {
            // save actual argument to temporary string
            std::string actualArg = std::string(this->argv[i]);
            // convert it to lwoer case
            std::transform(actualArg.begin(), actualArg.end(), actualArg.begin(), ::tolower);

            if (actualArg == "-w") { // show only original and cropped image
                this->enableWindows = false;
            }
            if (actualArg == "-suh" || actualArg == "suh") { // run Suh's auto cropping methods
                this->suh = true;
            }
            if (actualArg == "-sten" || actualArg == "sten" || actualArg == "stentiford") { // run Stentiford's auto cropping methods
                this->stentiford = true;
            }
            if (actualArg == "-fang" || actualArg == "fang") { // run Fang's auto cropping methods
                this->fang = true;
            }


            // method for cropping ROI with specified width and height
            if (actualArg == "-wh" || actualArg == "wh") { 
                this->wh = true; // set flag

                if (this->argc > (i + 1)) { // save width value
                    char *endptr;
                    this->width = strtol(this->argv[i+1], &endptr, 10); // check if it is valid number
                    if (*endptr != '\0') {
                        this->allClear = false;
                        std::cerr << "Invalid argument of width!" << std::endl;
                        return;
                    }
                }
                if (this->argc > (i + 2)) { // save height value
                    char *endptr;
                    this->height = strtol(this->argv[i+2], &endptr, 10); // check if it is valid number
                    if (*endptr != '\0') {
                        this->allClear = false;
                        std::cerr << "Invalid argument of height!" << std::endl;
                        return;
                    }
                }

                // check if values were defined successfully
                if (this->width <= 0 || this->height <= 0) {
                    this->allClear = false;
                    std::cerr << "Invalid or undefined arguments of width or height!" << std::endl;
                    return;
                }
            }
            else if (actualArg == "-scale" || actualArg == "scale") { // method for cropping ROI with specified scale factor
                this->scale = true; // set flag

                // save scale factor value
                if (this->argc > (i + 1)) {
                    char *endptr;
                    this->scaleValue = (float)strtod(this->argv[i+1], &endptr);
                    if (*endptr != '\0') {
                        this->allClear = false;
                        std::cerr << "Invalid argument of scale value!" << std::endl;
                        return;
                    }
                }

                // check if value was defined successfully
                if (this->scaleValue <= 0.0f || this->scaleValue >= 1.0f) {
                    this->allClear = false;
                    std::cerr << "Invalid or undefined argument of scale value! It should be float in range (0;1)." << std::endl;
                    return;
                }
            }
            else if (actualArg == "-whratio" || actualArg == "whratio") { // method for cropping ROI with specified aspect ratio
                this->whRatio = true; // set flag

                // save width value in aspect ratio
                if (this->argc > (i + 1)) {
                    char *endptr;
                    this->wRatio = strtol(this->argv[i+1], &endptr, 10); // check if it is valid number
                    if (*endptr != '\0') {
                        this->allClear = false;
                        std::cerr << "Invalid argument of width ratio!" << std::endl;
                        return;
                    }
                }
                // save height value in aspect ratio
                if (this->argc > (i + 2)) {
                    char *endptr;
                    this->hRatio = strtol(this->argv[i+2], &endptr, 10); // check if it is valid number
                    if (*endptr != '\0') {
                        this->allClear = false;
                        std::cerr << "Invalid argument of height ratio!" << std::endl;
                        return;
                    }
                }

                // check if values were defined successfully
                if (this->wRatio <= 0 || this->hRatio <= 0) {
                    this->allClear = false;
                    std::cerr << "Invalid or undefined arguments of width or height ratios!" << std::endl;
                    return;
                }
            }
            else if (actualArg == "-threshold" || actualArg == "threshold") { // Suh's threshold value is specified
                this->threshold = true; // set flag

                // save saliency threshold value
                if (this->argc > (i + 1)) {
                    char *endptr;
                    this->suhThreshold = (float)strtod(this->argv[i+1], &endptr);
                    if (*endptr != '\0') {
                        this->allClear = false;
                        std::cerr << "Invalid argument of threshold for Suh automatic cropping algorithm!" << std::endl;
                        return;
                    }
                }

                // check if threshold value was defined successfully
                if (this->suhThreshold <= 0.0f || this->suhThreshold >= 1.0f) {
                    this->allClear = false;
                    std::cerr << "Invalid or undefined argument of saliency threshold for Suh automatic cropping algorithm!" << std::endl;
                    return;
                }
            }
        }
    }

    // arguments -wh -scale -whratio cannot be combined
    if ((this->wh && this->scale) || (this->wh && this->whRatio) || (this->scale && this->whRatio)) {
        std::cerr << "Invalid combination of arguments(wh,scale,whRatio)!" << std::endl;
        this->allClear = false;
    }
}


/**
 * Getter function to indicate if there was no error in parsing arguments
 * @return True if there was no error, else False
 */
bool Arguments::isAllClear() {
    return this->allClear;
}


/**
 * Getter function to indicate if help was requested
 * @return True if there was -help argument, else False
 */
bool Arguments::isHelpActivated() {
    return this->help;
}

/**
 * Getter function to indicate if all windows should be shown
 * @return False if argument -w was specified, else True
 */
bool Arguments::isWindowsEnabled() {
    return this->enableWindows;
}

/**
 * Getter function to indicate if Suh's methods should be run
 * @return True to run Suh's method, else False
 */
bool Arguments::isSuh() {
    return this->suh;
}

/**
 * Setter function to run Suh's methods
 */
void Arguments::setSuh() {
    this->suh = true;
}

/**
 * Getter function to indicate if Stentiford's methods should be run
 * @return True to run Stentiford's method, else False
 */
bool Arguments::isStentiford() {
    return this->stentiford;
}

/**
 * Setter function to run Suh's methods
 */
void Arguments::setStentiford() {
    this->stentiford = true;
}

/**
 * Getter function to indicate if Fang's methods should be run
 * @return True to run Fang's method, else False
 */
bool Arguments::isFang() {
    return this->fang;
}

/**
 * Setter function to run Fang's methods
 */
void Arguments::setFang() {
    this->fang = true;
}

/**
 * Getter function to indicate if cropping method with specified width and height should be run
 * @return True to run width and height cropping method, else False
 */
bool Arguments::isWH() {
    return this->wh;
}

/**
 * Getter function to indicate if cropping method with specified scale factor should be run
 * @return True to run scaling method, else False
 */
bool Arguments::isScale() {
    return this->scale;
}

/**
 * Getter function to indicate if cropping method with specified aspect ratio should be run
 * @return True to run aspect ratio cropping method, else False
 */
bool Arguments::isWHratio() {
    return this->whRatio;
}

/**
 * Getter function to indicate if threshhold is specified
 * @return True if threshold is specified, else False
 */
bool Arguments::isThreshold() {
    return this->threshold;
}

/**
 * Getter function for width of cropping ROI
 * @return Width of ROI
 */
int Arguments::getWidth() {
    return this->width;
}

/**
 * Getter function for height of cropping ROI
 * @return Height of ROI
 */
int Arguments::getHeight() {
    return this->height;
}

/**
 * Getter function for scale factor (area of cropping ROI : area of original image)
 * @return Scale factor
 */
float Arguments::getScale() {
    return this->scaleValue;
}

/**
 * Getter function for width value in aspect ratio
 * @return Width value in aspect ratio
 */
int Arguments::getWidthRatio() {
    return this->wRatio;
}

/**
 * Getter function for height value in aspect ratio
 * @return Height value in aspect ratio
 */
int Arguments::getHeightRatio() {
    return this->hRatio;
}

/**
 * Getter function for threshold value used in Suh's cropping methods
 * @return Threshold value
 */
float Arguments::getThreshold() {
    return this->suhThreshold;
}