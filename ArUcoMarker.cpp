#include "ArUcoMarker.h"
#include <iostream>

int ArUcoMarker::markersOfBoardDetected;

//function for reading camera parameter
// read camera calibration data
bool ArUcoMarker::readParameters(string filename, int screenWidth, int screenHeight) {
    frameWidth = screenWidth;
    frameHeight = screenHeight;
    numofMarketThreshold = 0;
    
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
    {
        cerr << "Invalid marker settings" << endl;
        return false;
    }
    fs["markersX"] >> markersX;
    fs["markersY"] >> markersY;
    fs["markerLength"] >> markerLength;
    fs["markerSeparation"] >> markerSeparation;
    fs["dictionaryId"] >> dictionaryId;
    fs["showRejected"] >> showRejected;
    fs["refindStrategy"] >> refindStrategy;
    fs["halfSideLengthDivRadius"] >> halfSideLengthDivRadius;
    fs["calibrationFile"] >> calibrationFile;
    fs["detectorFile"] >> detectorFile; //can be null string
    fs["numofMarketThreshold"] >> numofMarketThreshold;


    // readCameraParameters
    if (!calibrationFile.empty())
    {
        FileStorage fs_cam(calibrationFile, FileStorage::READ);
        if (!fs_cam.isOpened())
        {
            cerr << "Invalid camera parameter file" << endl;
            return false;
        }
        fs_cam["camera_matrix"] >> camMatrix;
        fs_cam["distortion_coefficients"] >> distCoeffs;
    }


    // readMarkerDetectorParameters
    detectorParams = aruco::DetectorParameters::create();
    if (!detectorFile.empty())
    {
        FileStorage fs_det(detectorFile, FileStorage::READ);
        if (!fs_det.isOpened())
        {
            cerr << "Invalid detector parameters file" << endl;
            return false;
        }

        fs_det["adaptiveThreshWinSizeMin"] >> detectorParams->adaptiveThreshWinSizeMin;
        fs_det["adaptiveThreshWinSizeMax"] >> detectorParams->adaptiveThreshWinSizeMax;
        fs_det["adaptiveThreshWinSizeStep"] >> detectorParams->adaptiveThreshWinSizeStep;
        fs_det["adaptiveThreshConstant"] >> detectorParams->adaptiveThreshConstant;
        fs_det["minMarkerPerimeterRate"] >> detectorParams->minMarkerPerimeterRate;
        fs_det["maxMarkerPerimeterRate"] >> detectorParams->maxMarkerPerimeterRate;
        fs_det["polygonalApproxAccuracyRate"] >> detectorParams->polygonalApproxAccuracyRate;
        fs_det["minCornerDistanceRate"] >> detectorParams->minCornerDistanceRate;
        fs_det["minMarkerDistanceRate"] >> detectorParams->minMarkerDistanceRate;
        fs_det["minDistanceToBorder"] >> detectorParams->minDistanceToBorder;
        fs_det["markerBorderBits"] >> detectorParams->markerBorderBits;
        fs_det["minOtsuStdDev"] >> detectorParams->minOtsuStdDev;
        fs_det["perspectiveRemovePixelPerCell"] >> detectorParams->perspectiveRemovePixelPerCell;
        fs_det["perspectiveRemoveIgnoredMarginPerCell"] >> detectorParams->perspectiveRemoveIgnoredMarginPerCell;
        fs_det["maxErroneousBitsInBorderRate"] >> detectorParams->maxErroneousBitsInBorderRate;
        fs_det["errorCorrectionRate"] >> detectorParams->errorCorrectionRate;
        fs_det["doCornerRefinement"] >> detectorParams->doCornerRefinement;
        fs_det["cornerRefinementWinSize"] >> detectorParams->cornerRefinementWinSize;
        fs_det["cornerRefinementMaxIterations"] >> detectorParams->cornerRefinementMaxIterations;
        fs_det["cornerRefinementMinAccuracy"] >> detectorParams->cornerRefinementMinAccuracy;
    }
    detectorParams->doCornerRefinement = true; // do corner refinement in markers

    return true;
}

void ArUcoMarker::initialize()
{
    inv_camMat = camMatrix.inv();

    // initialize aruco marker
    dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));
    // create board object
    gridboard = aruco::GridBoard::create(markersX, markersY, markerLength, markerSeparation, dictionary);
    board = gridboard.staticCast<aruco::Board>();

    // init sphere setting
    boardX = (float)(markersX * markerLength + (markersX - 1) * markerSeparation);
    boardY = (float)(markersY * markerLength + (markersY - 1) * markerSeparation); // ex: 30cm
    axisLength = boardX * halfSideLengthDivRadius * 0.5f; // ex: 15cm -> 10cm

    markersOfBoardDetected = 0;
}

cv::Point2f ArUcoMarker::backproject3DPoint(const cv::Point3f &point3d)
{
    // 3D point vector [x y z 1]'
    cv::Mat point3d_vec = cv::Mat(4, 1, CV_64FC1);
    point3d_vec.at<double>(0) = point3d.x;
    point3d_vec.at<double>(1) = point3d.y;
    point3d_vec.at<double>(2) = point3d.z;
    point3d_vec.at<double>(3) = 1;

    // 2D point vector [u v 1]'
    cv::Mat point2d_vec = cv::Mat(4, 1, CV_64FC1);
    point2d_vec = camMatrix * _P_matrix * point3d_vec;

    // Normalization of [u v]'
    cv::Point2f point2d;
    point2d.x = (float)(point2d_vec.at<double>(0) / point2d_vec.at<double>(2));
    point2d.y = (float)(point2d_vec.at<double>(1) / point2d_vec.at<double>(2));

    return point2d;
}

void ArUcoMarker::set_P_matrix(const cv::Mat &R_matrix, const cv::Mat &t_matrix)
{
    // Rotation-Translation Matrix Definition
    _P_matrix.at<double>(0, 0) = R_matrix.at<double>(0, 0);
    _P_matrix.at<double>(0, 1) = R_matrix.at<double>(0, 1);
    _P_matrix.at<double>(0, 2) = R_matrix.at<double>(0, 2);
    _P_matrix.at<double>(1, 0) = R_matrix.at<double>(1, 0);
    _P_matrix.at<double>(1, 1) = R_matrix.at<double>(1, 1);
    _P_matrix.at<double>(1, 2) = R_matrix.at<double>(1, 2);
    _P_matrix.at<double>(2, 0) = R_matrix.at<double>(2, 0);
    _P_matrix.at<double>(2, 1) = R_matrix.at<double>(2, 1);
    _P_matrix.at<double>(2, 2) = R_matrix.at<double>(2, 2);
    _P_matrix.at<double>(0, 3) = t_matrix.at<double>(0);
    _P_matrix.at<double>(1, 3) = t_matrix.at<double>(1);
    _P_matrix.at<double>(2, 3) = t_matrix.at<double>(2);
}

std::vector<cv::Point2f> ArUcoMarker::markerDetect(InputArray _src)
{
    Mat src = _src.getMat();
    CV_Assert(src.type() == CV_8UC3 && src.size() == Size(frameWidth, frameHeight));

    vector< int > ids;
    vector< vector< Point2f > > corners, rejected;
    Vec3d rvec, tvec;

    //TODO:marker
    std::vector<cv::Point2f> bounding;
    cv::Point3f corner3d[4] = { Point3f(0.0, 0.0, 0.0),  // id=2
                                    Point3f(boardX, 0.0, 0.0), // id=3, red line is +x
                                    Point3f(0.0, boardY, 0.0), // id=0, green line is +y
                                    Point3f(boardX, boardY, 0.0) };// id=1

    // detect markers
    aruco::detectMarkers(src, dictionary, corners, ids, detectorParams, rejected);

    // refind strategy to detect more markers
    if (refindStrategy)
        aruco::refineDetectedMarkers(src, board, corners, ids, rejected, camMatrix,
        distCoeffs);

    // estimate board pose
    if (ids.size() >= numofMarketThreshold) {
        markersOfBoardDetected = aruco::estimatePoseBoard(corners, ids, board, camMatrix, distCoeffs, rvec, tvec);
    }
    else {
        markersOfBoardDetected = 0;
    }

    // draw results
    if (ids.size() > 0) {aruco::drawDetectedMarkers(src, corners, ids);}

    if (showRejected && rejected.size() > 0)
        aruco::drawDetectedMarkers(src, rejected, noArray(), Scalar(100, 0, 255));

    if (markersOfBoardDetected > 0)
    {
        // draw marker board 3 axis
        aruco::drawAxis(src, camMatrix, distCoeffs, rvec, tvec, axisLength);

        std::cout << "\nMarker per frame" << endl;
        int i = 0;
        for (std::vector<vector< Point2f >>::iterator it = corners.begin(); it != corners.end(); ++it, ++i) {
            std::cout << "ID : " << ids[i] << endl;
            int j = 0;
            for (std::vector<Point2f>::iterator jt = it->begin(); jt != it->end(); ++jt, ++j)
            {
                std::cout << jt->x << " / " << jt->y << endl;
            }
            std::cout << '*' << endl;
        } // TODO:no use

        // calculate viewMatirx
        Mat rotation = Mat(3, 3, CV_64F, Scalar(0.f));
        Rodrigues(rvec, rotation);

        Mat tMat = Mat(3, 1, CV_64FC1, &tvec);
        
        // TODO:marker BUG:not accurate
        cout << "\nOpneCV corners Round: " << endl;
        set_P_matrix(rotation, tMat);

        for (int i = 0; i < 4; i++)
        {
            Point2f temp = backproject3DPoint(corner3d[i]); // TODO:need elements clamp range!!
            bounding.push_back(temp);
            rectangle(src, temp - Point2f(3, 3),
                      temp + Point2f(3, 3), Scalar(255, 255, 0), 1, LINE_AA);
            cout << temp << endl;
        }
    }
    imshow("out", src);
    return bounding;
}

ArUcoMarker::~ArUcoMarker()
{
    // TODO destructor
}

