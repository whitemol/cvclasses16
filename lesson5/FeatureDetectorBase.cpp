///@File: FeatureDetectorBase.cpp
///@Brief: Contains implementation of interface for FeatureDetector classes
///@Author: Stepan Sidorov
///@Date: 01 November 2015

#include "stdafx.h"
#include "FeatureDetectorBase.h"
#include "FeatureDetectorHarris.h"
//#include "FeatureDetectorFAST.h"
#include "FeatureDetectorLoG.h"

FeatureDetectorBase* FeatureDetectorBase::CreateAlgorithm(const std::string& algorithmName)
{
    if (algorithmName == "Harris")
    {
        return new FeatureDetectorHarris();
    }
    //else if (algorithmName == "FAST")
    //{
    //    return new FeatureDetectorFAST();
    //}
    else if (algorithmName == "LoG")
    {
        return new FeatureDetectorLoG();
    }
    else
    {
        return nullptr;
    }
}
