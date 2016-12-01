///@File: FeatureDetectorHarris.cpp
///@Brief: Contains implementation of FeatureDetectorHarris class
///@Author: Stepan Sidorov
///@Date: 01 November 2015

#pragma once

#include "stdafx.h"
#include "FeatureDetectorHarris.h"

FeatureDetectorHarris::FeatureDetectorHarris()
{
    m_param = { 5, 4, 100 };
}

void FeatureDetectorHarris::Run(const cv::Mat &img)
{
    m_param.srcImage = img;

    // Create DEMO window
    m_param.windowName = GetName();
    cv::namedWindow(m_param.windowName, CV_WINDOW_AUTOSIZE);

    cv::createTrackbar("WinSize",
        m_param.windowName, &m_param.windowSize, 7, findFeatures, static_cast<void*>(&m_param));
    cv::createTrackbar("k * 100",
        m_param.windowName, &m_param.k, 20, findFeatures, static_cast<void*>(&m_param));
    cv::createTrackbar("Thresh",
        m_param.windowName, &m_param.threshold, 200, findFeatures, static_cast<void*>(&m_param));

    cv::waitKey(0);
    cv::destroyWindow(m_param.windowName);
}

void FeatureDetectorHarris::findFeatures(int pos, void *data)
{
    const Params& userData = *static_cast<Params*>(data);

    if (userData.windowSize % 2 == 0)
    {
        // Skip invalid values
        return;
    }

    cv::Mat gray, dst, dstNorm, show;
    cv::cvtColor(userData.srcImage, gray, CV_BGR2GRAY);
    userData.srcImage.copyTo(show);

    // Detect corners
    int sobelKernelSize = 3;
    cornerHarris(gray, dst, userData.windowSize, sobelKernelSize, static_cast<double>(userData.k) / 100, cv::BORDER_DEFAULT);

    // Normalize
    cv::normalize(dst, dstNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    // Draw feature points
    for (int i = 0; i < dstNorm.rows; i++)
    {
        for (int j = 0; j < dstNorm.cols; j++)
        {
            if (dstNorm.at<float>(i, j) > userData.threshold)
            {
                cv::rectangle(show, cv::Point(j - 1, i - 1), cv::Point(j + 1, i + 1), cv::Scalar(0, 0, 255));
            }
        }
    }

    // Show result
    cv::imshow(userData.windowName, show);
}
