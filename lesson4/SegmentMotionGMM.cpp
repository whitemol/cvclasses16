///@File: SegmentMotionGMM.cpp
///@Brief: Contains implementation of segmentation based on gaussian mixture
///        model
///@Author: Vitaliy Baldeev
///@Date: 04 October 2015

#include "SegmentMotionGMM.h"

#include <iostream>

#include <opencv2\videoio\videoio.hpp>
#include "opencv2\highgui.hpp"

///////////////////////////////////////////////////////////////////////////////
cv::Mat SegmentMotionGMM::process(cv::VideoCapture& capture)
{
    if(!m_algorithmPtr)
    {
        m_algorithmPtr = cv::createBackgroundSubtractorMOG2();
    }

    cv::Mat currentFrame;
    capture >> currentFrame;

    cv::Mat result;
    m_algorithmPtr->apply(currentFrame, result);
    return result;
}

///////////////////////////////////////////////////////////////////////////////
void SegmentMotionGMM::createGUI()
{
    const std::string windowName = GetName();
    cv::namedWindow(windowName);

    m_params.learningRate = 30;
    m_params.history = 30;
    m_params.varThreshold = 30;

    cv::createTrackbar("Learning Rate", windowName, &m_params.learningRate, 100);
    cv::createTrackbar("History", windowName, &m_params.history, 1000);
    cv::createTrackbar("Var Threshold", windowName, &m_params.varThreshold, 255);
}
