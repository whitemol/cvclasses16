///@File: FeatureDetectorHarris.h
///@Brief: Contains declaration of FeatureDetectorHarris class
///@Author: Stepan Sidorov
///@Date: 01 November 2015

#pragma once

#include "stdafx.h"
#include "FeatureDetectorBase.h"

class FeatureDetectorHarris : public FeatureDetectorBase
{
public:
    ///@brief Constructor
    FeatureDetectorHarris();

    ///@see FeatureDetectorBase::Run
    virtual void Run(const cv::Mat &img);

    ///@see FeatureDetectorBase::GetName
    virtual std::string GetName() const override
    {
        return "Harris detector";
    }

protected:
    ///@brief find feature points
    static void findFeatures(int pos, void *data);
    
    ///@brief structure of parameters with single object - m_param
    struct Params
    {
        int windowSize;         ///< Size of window
        int k;                  ///< Value of parameter k
        int threshold;          ///< Value of threshold

        cv::String windowName;  ///< Name of window
        cv::Mat srcImage;       ///< Image to be proccessed
    } m_param;
};
