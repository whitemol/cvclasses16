///@File: ObjectTrackingLK.h
///@Brief: declaration of ObjectTrackingLK class
///@Author: Sidorov Stepan
///@Date: 07.12.2015

#include "stdafx.h"
#include "IObjectTracking.h"

#pragma once

class ObjectTrackingLK : public IObjectTracking
{
public:
    ///@see IObjectTracking::Run
    virtual void Run(cv::VideoCapture &capture);

    ///@brife reply name
    virtual std::string GetName() const
    {
        return "Lucas-Kanade algorythm";
    }

private:
    ///@brief help function
    void help();

    ///@brief mouse callback function
    static void onMouse(int event, int x, int y, int, void*);
};
