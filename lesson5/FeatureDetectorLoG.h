///@File: FeatureDetectorLoG.h
///@Brief: Contains declaration of FeatureDetectorLoG class
///@Author: Stepan Sidorov
///@Date: 02 November 2015

#pragma once

#include "stdafx.h"
#include "FeatureDetectorBase.h"

class FeatureDetectorLoG : public FeatureDetectorBase
{
public:
    ///@brief Constructor
    FeatureDetectorLoG();

    ///@see FeatureDetectorBase::Run
    virtual void Run(const cv::Mat &img);

    ///@see FeatureDetectorBase::GetName
    virtual std::string GetName() const override
    {
        return "LoG detector";
    }

protected:
    ///@brief find feature points
    static void findFeatures(int pos, void *data);

    ///@ Àunction that defines whether a point is a local maximum in 3D LoG array
    static bool isLocalMax(float *LoG, int row, int col, int n, int i, int j);

    ///@brief structure of parameters with single object - m_param
    struct Params
    {
        int layersNum;          ///< Number of layers   
        int sigma0;             ///< Initial value of sigma
        int sigmaStep;          ///< Value of step: sigma(n) = sigma(n-1)*step
        int threshold;          ///< Value of threshold

        cv::String windowName;  ///< Name of window
        cv::Mat srcImage;       ///< Image to be proccessed
    } m_param;
};
