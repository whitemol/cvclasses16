///@File: SegmentMotionBU.h
///@Brief: Contains implementation of segmentation based on difference
///        between two frames and background updating
///@Author: Vitaliy Baldeev
///@Date: 03 October 2015

#pragma once

#include "SegmentMotionBase.h"

#include "opencv2\core\mat.hpp"

///@class SegmentMotionBU
/// Demonstrates the algorithm of simplest background subtraction with
/// background updating per every frame
class SegmentMotionBU : public SegmentMotionBase
{
public:
    ///@brief ctor
    SegmentMotionBU()
        : m_prevBackgroundUpdated(false)
    {
    }

    ///@see SegmentMotionBase::GetName
    virtual std::string GetName() const override
    {
        return "SegmentMotionBU";
    }

protected:
    ///@see SegmentMotionBase::process
    virtual cv::Mat process(cv::VideoCapture& capture) override;

    ///@see SegmentMotionBase::createGUI
    virtual void createGUI() override;

    ///@brief Update background
    void updateBackground(cv::Mat& currentFrame);

    ///@brief structure of parameters with single object - m_params
    struct Params
    {
        int threshold;          ///< maximum of distance between background and current frame
        int alpha;              ///< velocity of learning
    };

    Params m_params;
    cv::Mat m_prevBackground;   ///< previous backround image
    bool m_prevBackgroundUpdated;
};
