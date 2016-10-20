///@File: SegmentMotion1G.cpp
///@Brief: Contains implementation of segmentation based on One Gaussian
///@Author: Kuksova Svetlana
///@Date: 26 October 2015

#include "SegmentMotion1G.h"

#include <iostream>
#include <iterator>

#include "opencv2\video\video.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"

///////////////////////////////////////////////////////////////////////////////
cv::Mat SegmentMotion1G::process(cv::VideoCapture& capture)
{
    if (m_params.historySize == 0)
    {
        m_params.historySize = 1;
    }

    cv::Mat frame;

    if (m_frameBuffer.size() < m_params.historySize)
    {
        while (m_frameBuffer.size() < m_params.historySize)
        {
            capture >> frame;
            cv::cvtColor(frame, frame, CV_BGR2GRAY);
            m_frameBuffer.push_back(frame);
        }
    }
    if (m_frameBuffer.size() > m_params.historySize)
    {
        while (m_frameBuffer.size() > m_params.historySize)
        {
            m_frameBuffer.pop_front();
        }
    }

    capture >> frame;
    cv::cvtColor(frame, frame, CV_BGR2GRAY);
    m_frameBuffer.pop_front();
    m_frameBuffer.push_back(frame);

    //Calculate m, sigma
    m_mMat = cv::Mat_<float>(frame.rows, frame.cols, 0.0);
    m_sigmaMat = cv::Mat_<float>(frame.rows, frame.cols, 0.0);


    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            std::list<cv::Mat>::const_iterator pos;

            int n = 0;
            for (pos = m_frameBuffer.begin(); pos != m_frameBuffer.end(); pos++)
            {
                const float val = static_cast<float>((*pos).at<uchar>(i, j));
                n++;

                // Calculate m
                float sum = 0;
                for (int k = 1; k <= n; k++)
                {
                    sum = sum + val;
                }
                m_mMat(i, j) = 1 / n*sum;

                //Calculate sigma
                float sum1 = 0;
                for (int k = 1; k <= n; k++)
                {
                    sum1 = sum1 + (val - 1 / n*sum)*(val - 1 / n*sum);
                }

                m_sigmaMat(i, j) = 1 / n*sum1;
            }
        }
    }

    // Detect foreground
    cv::Mat result(frame.rows, frame.cols, CV_8UC1);
    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            if ((abs(m_mMat(i, j) - static_cast<float>(frame.at<uchar>(i, j))) / m_sigmaMat(i, j)) >
                static_cast<float>(m_params.T))
            {
                result.at<uchar>(i, j) = static_cast<uchar>(0);
            }
            else
            {
                result.at<uchar>(i, j) = static_cast<uchar>(255);
            }
        }
    }

    return result;
}

///////////////////////////////////////////////////////////////////////////////
void SegmentMotion1G::createGUI()
{
    const std::string windowName = GetName();
    cv::namedWindow(windowName);

    m_params.historySize = 10;
    m_params.T = 3;

    cv::createTrackbar("History", windowName, reinterpret_cast<int*>(&m_params.historySize), 20);
    cv::createTrackbar("T", windowName, &m_params.T, 20);
}
