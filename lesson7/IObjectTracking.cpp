///@File: IObjectTracking.cpp
///@Brief: implementation of interface for ObjectTracking classes
///@Author: Sidorov Stepan
///@Date: 07.12.2015

#include "stdafx.h"
#include "IObjectTracking.h"
#include "ObjectTrackingLK.h"
#include "ObjectTrackingTK.h"
#include "ObjectTrackingSTK.h"

IObjectTracking* IObjectTracking::CreateAlgorythm(const std::string & name)
{
    if (name == "LK")
        return new ObjectTrackingLK();
    else if (name == "TK")
        return new ObjectTrackingTK();
    else if (name == "STK")
        return new ObjectTrackingSTK();
    else
        return nullptr;
}
