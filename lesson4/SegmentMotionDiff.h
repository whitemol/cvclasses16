///@File: SegmentMotionDiff.h
///@Brief: Contains implementation of simple segmentation based on difference between two frames
///@Author: Vitaliy Baldeev
///@Date: 01 October 2015

#pragma once

#include <iostream>

#include "SegmentMotionBase.h"

#include "opencv2\core\mat.hpp"

///@class SegmentMotionDiff
/// Demonstrates the algorithm of simplest background subtraction with
/// no background updating
class SegmentMotionDiff : public SegmentMotionBase
{
public:
    SegmentMotionDiff()
        : m_backgroundUpdated(false)
    {
    }

    ///@see SegmentMotionBase::GetName
    virtual std::string GetName() const override
    {
        return "SegmentMotionDiff";
    }

protected:
    ///@see SegmentMotionBase::process
    virtual cv::Mat process(cv::VideoCapture& capture) override;

    ///@see SegmentMotionBase::createGUI
    virtual void createGUI() override;

    ///@brief threshold of segmentation
    int m_threshold;

    bool m_backgroundUpdated;
    cv::Mat m_background;
};
