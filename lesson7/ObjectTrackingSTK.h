///@File: ObjectTrackingSTK.h
///@Brief: declaration of ObjectTrackingSTK class
///@Author: Sidorov Stepan and Kuksova Svetlana
///@Date: 20.12.2015

#include "stdafx.h"
#include "IObjectTracking.h"

#pragma once

class ObjectTrackingSTK : public IObjectTracking
{
public:
    ///@see IObjectTracking::Run
    virtual void Run(cv::VideoCapture &capture);

    ///@brife reply name
    virtual std::string GetName() const
    {
        return "Shi-Tomasi-Kanade algorythm";
    }

private:
    ///@brief help function
    void help();
};
