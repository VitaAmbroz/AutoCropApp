/**
 * Bachelor thesis: Algorithms for automatic image cropping
 * VUT FIT 2018
 * Author: Vít Ambrož (xambro15@stud.fit.vutbr.cz)
 * Supervisor: Doc. Ing. Martin Čadík, Ph. D.
 * File: Arguments.h
 * Github repository: https://github.com/VitaAmbroz/AutoCropApp
 */

#ifndef __ARGUMENTS_H__
#define __ARGUMENTS_H__

#include <string>
#include <iostream>
#include <algorithm>

using namespace std;

class Arguments {
public:
    // constructor
    Arguments(int mArgc, char** mArgv);
    // path of original image that will be cropped
    std::string imgPath;

    // getters and setters of flags and values
    bool isHelpActivated();
    bool isAllClear();
    bool isWindowsEnabled();

    bool isSuh();
    void setSuh();
    bool isStentiford();
    void setStentiford();
    bool isFang();
    void setFang();

    bool isWH();
    bool isScale();
    bool isWHratio();
    bool isThreshold();

    int getWidth();
    int getHeight();
    float getScale();
    int getWidthRatio();
    int getHeightRatio();
    float getThreshold();

private:
    void parse();
    int argc;       // argc count
    char** argv;    // reference for argv
    bool help;      // flag if help argument was set
    bool allClear;  // flag if parsing was succesfull
    bool enableWindows; // flag if all windows should be displayed

    bool suh;   // flag if Suh's algoritm should be used
    bool stentiford;  // flag if Stentiford's algorithm should be used
    bool fang;  // flag if Fang's algorithm should be used

    bool wh;    // flag if method with scpefied width and height of ROI should be run
    bool scale; // flag if method with specified scale factor should be run
    bool whRatio; // flag if method with specified aspect ratio should be run 
    bool threshold; // flag if threshold used in Suh's algorithm is specified

    int width;  // width of cropped ROI
    int height; // height of cropped ROI
    float scaleValue; // value of scale factor
    int wRatio; // width value in aspect ratio
    int hRatio; // height value in aspect ratio
    float suhThreshold; // value of saliency threshold
};


#endif //__ARGUMENTS_H__