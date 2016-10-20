///@File: SegmentMotionMinMax.h
///@Brief: Contains declaration of SegmentMotionMinMax class
///@Author: Stepan Sidorov
///@Date: 17 October 2015

#pragma once

#include <iostream>
#include <list>

#include "opencv2\core\mat.hpp"

#include "SegmentMotionBase.h"

///@class SegmentMotionMinMax
class SegmentMotionMinMax : public SegmentMotionBase
{
public:
    SegmentMotionMinMax() {}

    ///@see SegmentMotionBase::GetName
    virtual std::string GetName() const override
    {
        return "SegmentMotionMinMax";
    }

protected:
    ///@see SegmentMotionBase::process
    virtual cv::Mat process(cv::VideoCapture& capture) override;

    ///@see SegmentMotionBase::createGUI
    virtual void createGUI() override;

    ///@ brief parameters of algorythm
    struct Params
    {
        size_t historySize; ///@ size of history
        int tau;            ///@ value of parameter tau
    };

    Params m_params;
    
    ///@brief frame buffer
    std::list<cv::Mat> m_frameBuffer;
    ///@brief matrix of min value of each pixel for the history
    cv::Mat_<float> m_minMat;
    ///@brief matrix of max value of each pixel for the history
    cv::Mat_<float> m_maxMat;
    ///@brief matrix of a maximum of consecutive frames difference observed over a training sequence
    cv::Mat_<float> m_Dmat;
};
