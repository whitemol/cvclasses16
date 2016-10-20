///@File: SegmentMotionMinMax.cpp
///@Brief: Contains implementation of SegmentMotionMinMax class
///@Author: Stepan Sidorov
///@Date: 17 October 2015

#include <iterator>

#include "opencv2\videoio.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\highgui.hpp"

#include "SegmentMotionMinMax.h"

///////////////////////////////////////////////////////////////////////////////
cv::Mat SegmentMotionMinMax::process(cv::VideoCapture& capture)
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

    // Calculate min, max, and D
    m_minMat = cv::Mat_<float>(frame.rows, frame.cols, 255.0);
    m_maxMat = cv::Mat_<float>(frame.rows, frame.cols, 0.0);
    m_Dmat = cv::Mat_<float>(frame.rows, frame.cols, 20.0);
    
    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            std::list<cv::Mat>::const_iterator pos;
            float previous = 0;

            for (pos = m_frameBuffer.begin(); pos != m_frameBuffer.end(); pos++)
            {
                const float val = static_cast<float>((*pos).at<uchar>(i, j));

                // Calculate max
                if (val > m_maxMat(i, j))
                {
                    m_maxMat(i, j) = val;
                }
                // Calculate min
                if (val < m_minMat(i, j))
                {
                    m_minMat(i, j) = val;
                }
                // Calculate D
                if (pos != m_frameBuffer.begin())
                {
                    float absDiff = abs(val - previous);
                    previous = val;

                    if (absDiff > m_Dmat(i, j))
                    {
                        m_Dmat(i, j) = absDiff;
                    }
                }
            }
        }
    }

    // Calculate median of D
    const int vSize = m_Dmat.rows * m_Dmat.cols;
    std::vector<float> vecD(vSize);
    for (int i = 0; i < m_Dmat.rows; i++)
    {
        for (int j = 0; j < m_Dmat.cols; j++)
        {
            vecD[i * m_Dmat.cols + j] = m_Dmat(i, j);
        }
    }
    std::sort(vecD.begin(), vecD.end());
    float median;
    if (vSize % 2 == 0)
    {
        median = (vecD[vSize / 2 - 1] + vecD[vSize / 2]) / 2;
    }
    else
    {
        median = vecD[vSize / 2];
    }

    // Detect foreground
    cv::Mat result(frame.rows, frame.cols, CV_8UC1);
    for (int i = 0; i < frame.rows; i++)
    {
        for (int j = 0; j < frame.cols; j++)
        {
            if (abs(m_maxMat(i, j) - static_cast<float>(frame.at<uchar>(i, j))) <
                static_cast<float>(m_params.tau) / 100 * median ||
                abs(m_minMat(i, j) - static_cast<float>(frame.at<uchar>(i, j))) <
                static_cast<float>(m_params.tau) / 100 * median)
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
void SegmentMotionMinMax::createGUI()
{
    const std::string windowName = GetName();
    cv::namedWindow(windowName);

    m_params.historySize = 10;
    m_params.tau = 5;

    cv::createTrackbar("History", windowName, reinterpret_cast<int*>(&m_params.historySize), 20);
    cv::createTrackbar("Tau * 100", windowName, &m_params.tau, 20);
}
