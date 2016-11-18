///@File: SegmentMotionMedian.cpp
///@Brief: Contains implementation of segmentation based on Median Filter
///@Author: Minaev Alexey
///@Date: 10 November 2016

#include "SegmentMotionMedian.h"

#include <iostream>
#include <iterator>

#include "opencv2\video\video.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"


///////////////////////////////////////////////////////////////////////////////
SegmentMotionMedian::SegmentMotionMedian()
{
	m_params.N = 10;
	m_params.T = 10;
}

///////////////////////////////////////////////////////////////////////////////
cv::Mat SegmentMotionMedian::process(cv::VideoCapture& capture)
{
	// validate parameters
	if (m_params.N < 1)
		m_params.N = 1;
	if (m_params.T < 1)
		m_params.T = 1;


	// read frame
	cv::Mat currentFrame;
	capture >> currentFrame;
	if (currentFrame.empty())
		return currentFrame;
	cv::cvtColor(currentFrame, currentFrame, CV_RGB2GRAY);


	// calculate background
	// very last frame is current, others - history frames
	m_history.push_back(currentFrame);
	while (m_history.size() - 1 > m_params.N)
	{
		m_history.pop_front();
	}
	int S = m_history.size();
	if (S == 1)
	{
		// first iteration
		return cv::Mat::zeros(currentFrame.size(), CV_8UC1);
	}
	cv::Mat background = cv::Mat::zeros(currentFrame.size(), CV_8UC1);
	if (S == 2)
	{
		// background is previuos frame
		m_history.begin()->copyTo(background);
	}
	if (S == 3)
	{
		// background is mean of prevoius two frames
		cv::Mat temp1, temp2;
		auto it = m_history.begin();
		it->convertTo(temp1, CV_32FC1);
		it++;
		it->convertTo(temp2, CV_32FC1);
		background = 0.5*temp1 + 0.5*temp2;
		background.convertTo(background, CV_8UC1);
	}
	if (S > 3)
	{
		// get median for each pixel
		for (int y = 0; y < currentFrame.size().height; y++)
		{
			for (int x = 0; x < currentFrame.size().width; x++)
			{
				uchar* hist = new uchar[S-1]; // histogram
				int i = 0;
				for (auto h = m_history.begin(); i < S - 1; h++, i++)
				{
					uchar val = (*h).at<char>(cv::Point(x, y));
					hist[i] = val;
				}
				uchar* unsorted = new uchar[S];
				memcpy(unsorted, hist, S);
				uchar median;
				if (S % 2)
				{
					// odd size
					std::nth_element(hist, hist + S / 2, hist + S);
					median = *(hist + S / 2);
				}
				else
				{
					// even size
					std::nth_element(hist, hist + S / 2, hist + S);
					std::nth_element(hist, hist + S / 2 - 1, hist + S);
					median = 0.5*double(hist[S / 2]) + 0.5*double(hist[S / 2 - 1]);
				}
				background.at<char>(cv::Point(x, y)) = median;
			}
		}
	}


	// extract foreground
	cv::Mat foreground;
	cv::absdiff(currentFrame, background, foreground);
	cv::threshold(foreground, foreground, m_params.T, 255, CV_THRESH_BINARY);

	return foreground;
}

///////////////////////////////////////////////////////////////////////////////
void SegmentMotionMedian::createGUI()
{
	const std::string windowName = GetName();
	cv::namedWindow(windowName);

	// m_params.historySize = 3;
	// m_params.threshold = 10;

	cv::createTrackbar("History size", windowName, reinterpret_cast<int*>(&m_params.N), 20);
	cv::createTrackbar("Threshold", windowName, &m_params.T, 20);
}
