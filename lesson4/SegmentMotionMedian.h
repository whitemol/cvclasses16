///@File: SegmentMotionMedian.h
///@Brief: Contains implementation of segmentation based on Median Filter
///@Author: Minaev Alexey
///@Date: 10 November 2016

#pragma once

#include <iostream>
#include <list>
#include <algorithm> // n-th element
#include "SegmentMotionBase.h"
#include "opencv2\core\mat.hpp"


///@class SegmentMotionMedian
/// Demonstrates the Median Filter algorithm of background subtraction
class SegmentMotionMedian : public SegmentMotionBase
{
public:
	SegmentMotionMedian();

	virtual std::string GetName() const override
	{
		return "SegmentMotionMedian";
	}

protected:
	virtual cv::Mat process(cv::VideoCapture& capture) override;

	virtual void createGUI() override;

	struct Params
	{
		size_t N; // history size
		int    T; // threshold
	} m_params;

	std::list<cv::Mat> m_history; // [frameN,...,frame1, frameCurrent]
};
