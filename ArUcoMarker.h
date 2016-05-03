#ifndef __ARUCOMARKER_H
#define __ARUCOMARKER_H

#pragma once
#include "stdafx.h"

using namespace cv;
using namespace std;

class ArUcoMarker
{
public:
    ArUcoMarker()
    {
        // constructor
        _P_matrix = cv::Mat::zeros(3, 4, CV_64FC1);
    }
    ~ArUcoMarker();
    bool readParameters(string file, int screenWidth, int screenHeight);
    void initialize();
    std::vector<cv::Point2f> markerDetect(InputArray _src);
    Point2f backproject3DPoint(const Point3f &point3d);
    void set_P_matrix(const cv::Mat &R_matrix, const cv::Mat &t_matrix);

private:
    int frameWidth;
    int frameHeight;
    
    // Marker settings
    int markersX;
    int markersY;
    float markerLength;
    float markerSeparation;
    int dictionaryId;
    bool showRejected;
    bool refindStrategy;
    float halfSideLengthDivRadius;
    string calibrationFile;
    string detectorFile;

    Ptr<aruco::DetectorParameters> detectorParams;
    Ptr<aruco::Dictionary> dictionary;
    Ptr<aruco::GridBoard> gridboard;
    Ptr<aruco::Board> board;
    Mat camMatrix, inv_camMat, distCoeffs;
    float axisLength;

    //TODO:marker
    float boardX;
    float boardY;
    Mat _P_matrix;

    // Frame draw sphere and axis
    static int markersOfBoardDetected;

    // Output png format
    stringstream ss;
    int numofMarketThreshold;
};
#endif