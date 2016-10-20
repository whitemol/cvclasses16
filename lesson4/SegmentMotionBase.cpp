///@File: ISegmentMotion.cpp
///@Brief: Contains implementation of interface for SegmentMotion classes
///@Author: Vitaliy Baldeev
///@Date: 12 October 2015

#include "SegmentMotionBase.h"

#include <iostream>

#include "opencv2\video\video.hpp"
#include "opencv2\highgui\highgui.hpp"

#include "SegmentMotionDiff.h"
#include "SegmentMotionBU.h"
#include "SegmentMotionGMM.h"
#include "SegmentMotionMinMax.h"
#include "SegmentMotion1G.h"

///////////////////////////////////////////////////////////////////////////////
void SegmentMotionBase::Run()
{
    cv::VideoCapture capture(0);

    if (!capture.isOpened())
    {
        std::cerr << "Can not open the camera !" << std::endl;
        exit(-1);
    }

    createGUI();

    while (true)
    {
        m_foreground = process(capture);
        cv::imshow(GetName(), m_foreground);

        if (cv::waitKey(1) >= 0)
        {
            break;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
SegmentMotionBase* SegmentMotionBase::CreateAlgorithm(std::string& algorithmName)
{
    if (algorithmName == "Diff")
    {
        return new SegmentMotionDiff();
    }
    else if (algorithmName == "BU")
    {
        return new SegmentMotionBU();
    }
    else if (algorithmName == "GMM")
    {
        return new SegmentMotionGMM();
    }
    else if (algorithmName == "MM")
    {
        return new SegmentMotionMinMax();
    }
	else if (algorithmName == "1G")
	{
		return new SegmentMotion1G();
	}
    else
    {
        return nullptr;
    }
}
