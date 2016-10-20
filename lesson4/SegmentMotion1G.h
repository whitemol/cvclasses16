///@File: SegmentMotion1G.h
///@Brief: Contains implementation of segmentation based on One Gaussian
///@Author: Kuksova Svetlana
///@Date: 26 October 2015

#pragma once

#include <iostream>
#include <list>
#include "SegmentMotionBase.h"

#include "opencv2\core\mat.hpp"
#include "opencv2\video\background_segm.hpp"

///@class SegmentMotion1G
/// Demonstrates the One gaussian algorithm of background subtraction
class SegmentMotion1G : public SegmentMotionBase
{
public:
    ///@brief ctor
    SegmentMotion1G() {}

    ///@see SegmentMotionBase::GetName
    virtual std::string GetName() const override
    {
        return "SegmentMotion1G";
    }

protected:
    ///@see SegmentMotionBase::process
    virtual cv::Mat process(cv::VideoCapture& capture) override;

    ///@see SegmentMotionBase::createGUI
    virtual void createGUI() override;

    ///@ brief parameters of algorythm
    struct Params
    {
        size_t historySize; ///< size of history
        int T;              ///< value of parameter T
    };

    Params m_params;

    ///@brief frame buffer
    std::list<cv::Mat> m_frameBuffer;
    ///@brief matrix of min value of each pixel for the history
    cv::Mat_<float> m_mMat;
    ///@brief matrix of max value of each pixel for the history
    cv::Mat_<float> m_sigmaMat;

    ///@brief Pointer to OpenCV algorithm of background subtraction
    //cv::Ptr<cv::BackgroundSubtractorMOG2> m_algorithmPtr;
};
