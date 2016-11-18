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
#include "SegmentMotionMedian.h"

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
void SegmentMotionBase::Run(std::string in_source_name)
{
	cv::VideoCapture source(in_source_name);

	if (!source.isOpened())
	{
		std::cerr << "Can not open " << in_source_name << std::endl;
		exit(-1);
	}

	int codec = static_cast<int>(source.get(CV_CAP_PROP_FOURCC));
	int width = source.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = source.get(CV_CAP_PROP_FRAME_HEIGHT);
	cv::VideoWriter writer("foreground.avi", codec, source.get(CV_CAP_PROP_FPS), cv::Size(width, height));

	if (!writer.isOpened())
	{
		std::cerr << "Can not open foreground.avi" << std::endl;
		exit(-1);
	}

	int frame_count = source.get(CV_CAP_PROP_FRAME_COUNT);

	while (true)
	{
		int current_frame = source.get(CV_CAP_PROP_POS_FRAMES);
		std::cout << current_frame << "/" << frame_count << std::endl;

		cv::Mat foreground = process(source);
		if (foreground.empty())
			break;

		cv::cvtColor(foreground, foreground, cv::COLOR_GRAY2BGR);
		writer << foreground;
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
	else if (algorithmName == "Med")
	{
		return new SegmentMotionMedian();
	}
    else
    {
        return nullptr;
    }
}
