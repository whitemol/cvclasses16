///@File: IObjectTracking.h
///@Brief: interface for ObjectTracking classes
///@Author: Sidorov Stepan
///@Date: 07.12.2015

#include "stdafx.h"

#pragma once

class IObjectTracking
{
public:
    ///@brief object tracking function
    virtual void Run(cv::VideoCapture &capture) = 0;

    ///@brief factory method
    static IObjectTracking* CreateAlgorythm(const std::string& algorithmName);

    ///@brief virtual destructor
    virtual ~IObjectTracking() {}

    ///@brife reply name
    virtual std::string GetName() const
    {
        return "ObjectTracking Base";
    }
};
