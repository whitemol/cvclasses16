///@File: FeatureDetectorDemo.cpp
///@Brief: Contains implementation for FeatureDetectorDemo class
///@Author: Stepan Sidorov
///@Date: 01 November 2015

#include "stdafx.h"
#include "FeatureDetectorDemo.h"

void FeatureDetectorDemo::Run(std::string algorithmName)
{
    std::unique_ptr<FeatureDetectorBase> pDetector(FeatureDetectorBase::CreateAlgorithm(algorithmName));
    if (!pDetector)
    {
        std::cout << "Invalid name of detector.\n";
        return;
    }

    cv::VideoCapture capture(0);
    cv::namedWindow("Video capture");
    cv::Mat frame;

    while (true)
    {
        capture >> frame;
        cv::imshow("Video capture", frame);
        char c = cv::waitKey(33);
        if (c == 27)
        {
            break;
        }
        if (c == 32)
        {
            pDetector->Run(frame);
        }
    }
}
