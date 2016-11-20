///@File: FeatureDetectorBase.h
///@Brief: Contains declaration of interface for FeatureDetector classes
///@Author: Stepan Sidorov
///@Date: 01 November 2015

#pragma once

#include "stdafx.h"

class FeatureDetectorBase
{
public:
    ///@brief launch demonstration
    virtual void Run(const cv::Mat &img) {}

    ///@brief factory method
    static FeatureDetectorBase* CreateAlgorithm(const std::string& algorithmName);

    ///@brief destructor
    virtual ~FeatureDetectorBase() {};

    ///@brief get the name of algorithm 
    virtual std::string GetName() const
    {
        return "Base";
    }

protected:
    ///@brief protected constructor
    FeatureDetectorBase() {}
};
