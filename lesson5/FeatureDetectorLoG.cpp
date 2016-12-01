///@File: FeatureDetectorLoG.cpp
///@Brief: Contains implementation of FeatureDetectorLoG class
///@Author: Stepan Sidorov
///@Date: 02 November 2015

#pragma once

#include "stdafx.h"
#include "FeatureDetectorLoG.h"

FeatureDetectorLoG::FeatureDetectorLoG()
{
    m_param = { 5, 10, 14, 10 };
}

void FeatureDetectorLoG::Run(const cv::Mat &img)
{
    m_param.srcImage = img;

    // Create DEMO window
    m_param.windowName = GetName();
    cv::namedWindow(m_param.windowName, CV_WINDOW_AUTOSIZE);

    cv::createTrackbar("LNum",
        m_param.windowName, &m_param.layersNum, 15, findFeatures, static_cast<void*>(&m_param));
    cv::createTrackbar("S0*10",
        m_param.windowName, &m_param.sigma0, 20, findFeatures, static_cast<void*>(&m_param));
    cv::createTrackbar("SStep*10",
        m_param.windowName, &m_param.sigmaStep, 20, findFeatures, static_cast<void*>(&m_param));
    cv::createTrackbar("Thresh",
        m_param.windowName, &m_param.threshold, 200, findFeatures, static_cast<void*>(&m_param));

    cv::waitKey(0);
    cv::destroyWindow(m_param.windowName);
}

void FeatureDetectorLoG::findFeatures(int pos, void *data)
{
    const Params& userData = *static_cast<Params*>(data);

    cv::Mat gray, blurred, laplacian, show;
    cv::cvtColor(userData.srcImage, gray, CV_BGR2GRAY);
    userData.srcImage.copyTo(show);
    int imgRows = gray.rows;
    int imgCols = gray.cols;

    // Use dynamic array because of higher adressing speed
    float *LoG = new float[userData.layersNum * imgRows * imgCols];
    float sigma0 = static_cast<float>(userData.sigma0) / 10;
    float sigma = sigma0;
    float sigmaStep = static_cast<float>(userData.sigmaStep) / 10;

    // Calculate 3D LoG array
    for (int n = 0; n < userData.layersNum; n++)
    {
        // Choose kernel size according to sigma value
        int kSize = static_cast<int>(sigma * 3 + 0.5) * 2 - 1;
        cv::GaussianBlur(gray, blurred, cv::Size(kSize, kSize), sigma);
        cv::Laplacian(blurred, laplacian, CV_32FC1);
        for (int i = 0; i < imgRows; i++)
        {
            for (int j = 0; j < imgCols; j++)
            {
                LoG[n*imgRows*imgCols + i*imgCols + j] = abs(laplacian.at<float>(i, j)) * sigma * sigma;
            }
        }
        sigma *= sigmaStep;
    }


    // Find feature feature points
    float sqrt2 = sqrtf(2);
    for (int n = 1; n < userData.layersNum - 1; n++)
    {
        for (int i = 1; i < imgRows - 1; i++)
        {
            for (int j = 1; j < imgCols - 1; j++)
            {
                if (LoG[n*imgRows*imgCols + i*imgCols + j] > userData.threshold && isLocalMax(LoG, imgRows, imgCols, n, i, j))
                {
                    cv::circle(show, cv::Point(j, i), static_cast<int>(sigma0 * pow(sigmaStep, n) * sqrt2), cv::Scalar(255, 255, 0));
                }
            }
        }
    }

    cv::imshow(userData.windowName, show);
    delete []LoG;
}

bool FeatureDetectorLoG::isLocalMax(float *LoG, int row, int col, int n, int i, int j)
{
    
    if (// Check neighbours from upper level
        LoG[n*row*col + i*col + j] > LoG[(n + 1)*row*col + (i - 1)*col + j - 1] &&
        LoG[n*row*col + i*col + j] > LoG[(n + 1)*row*col + (i - 1)*col + j] &&
        LoG[n*row*col + i*col + j] > LoG[(n + 1)*row*col + (i - 1)*col + j + 1] &&
        LoG[n*row*col + i*col + j] > LoG[(n + 1)*row*col + i*col + j - 1] &&
        LoG[n*row*col + i*col + j] > LoG[(n + 1)*row*col + i*col + j] &&
        LoG[n*row*col + i*col + j] > LoG[(n + 1)*row*col + i*col + j + 1] &&
        LoG[n*row*col + i*col + j] > LoG[(n + 1)*row*col + (i + 1)*col + j - 1] &&
        LoG[n*row*col + i*col + j] > LoG[(n + 1)*row*col + (i + 1)*col + j] &&
        LoG[n*row*col + i*col + j] > LoG[(n + 1)*row*col + (i + 1)*col + j + 1] &&
        // Check neighbours from current level
        LoG[n*row*col + i*col + j] > LoG[n*row*col + (i - 1)*col + j - 1] &&
        LoG[n*row*col + i*col + j] > LoG[n*row*col + (i - 1)*col + j] &&
        LoG[n*row*col + i*col + j] > LoG[n*row*col + (i - 1)*col + j + 1] &&
        LoG[n*row*col + i*col + j] > LoG[n*row*col + i*col + j - 1] &&
        LoG[n*row*col + i*col + j] > LoG[n*row*col + i*col + j + 1] &&
        LoG[n*row*col + i*col + j] > LoG[n*row*col + (i + 1)*col + j - 1] &&
        LoG[n*row*col + i*col + j] > LoG[n*row*col + (i + 1)*col + j] &&
        LoG[n*row*col + i*col + j] > LoG[n*row*col + (i + 1)*col + j + 1] &&
        // Check neighbours from lower level
        LoG[n*row*col + i*col + j] > LoG[(n - 1)*row*col + (i - 1)*col + j - 1] &&
        LoG[n*row*col + i*col + j] > LoG[(n - 1)*row*col + (i - 1)*col + j] &&
        LoG[n*row*col + i*col + j] > LoG[(n - 1)*row*col + (i - 1)*col + j + 1] &&
        LoG[n*row*col + i*col + j] > LoG[(n - 1)*row*col + i*col + j - 1] &&
        LoG[n*row*col + i*col + j] > LoG[(n - 1)*row*col + i*col + j] &&
        LoG[n*row*col + i*col + j] > LoG[(n - 1)*row*col + i*col + j + 1] &&
        LoG[n*row*col + i*col + j] > LoG[(n - 1)*row*col + (i + 1)*col + j - 1] &&
        LoG[n*row*col + i*col + j] > LoG[(n - 1)*row*col + (i + 1)*col + j] &&
        LoG[n*row*col + i*col + j] > LoG[(n - 1)*row*col + (i + 1)*col + j + 1])
    {
        return true;
    }
    else
    {
        return false;
    }
}
