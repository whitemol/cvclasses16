///@File: SegmentMotionGMM.h
///@Brief: Contains implementation of segmentation based on gaussian mixture
///        model
///@Author: Vitaliy Baldeev
///@Date: 04 October 2015

#pragma once

#include "SegmentMotionBase.h"

#include "opencv2\core\mat.hpp"

#include "opencv2\video\background_segm.hpp"

///@class SegmentMotionGMM
/// Demonstrates the gaussian mixture algorithm of background subtraction
class SegmentMotionGMM : public SegmentMotionBase
{
public:
    ///@brief ctor
    SegmentMotionGMM() {}

    ///@see SegmentMotionBase::GetName
    virtual std::string GetName() const override
    {
        return "SegmentMotionGMM";
    }

protected:
    ///@see SegmentMotionBase::process
    virtual cv::Mat process(cv::VideoCapture& capture) override;

    ///@see SegmentMotionBase::createGUI
    virtual void createGUI() override;

    ///@brief structure of parameters with single object - m_params
    struct Params
    {
        int learningRate;       ///< velocity of learning
        int history;            ///< how many frames take into account in computing of background
        int varThreshold;       ///< maximum of distance between background and current frame
    };

    Params m_params;

    ///@brief Pointer to OpenCV algorithm of background subtraction
    cv::Ptr<cv::BackgroundSubtractorMOG2> m_algorithmPtr;
};
