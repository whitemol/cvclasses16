///@File: ObjectTrackingTK.cpp
///@Brief: implementation of ObjectTrackingTK class
///@Author: Sidorov Stepan and Kuksova Svetlana
///@Date: 20.12.2015

#include "stdafx.h"
#include "ObjectTrackingTK.h"

void ObjectTrackingTK::Run(cv::VideoCapture & capture)
{
    help();

    cv::Mat frame, currentGray, prevGray;
    std::vector<cv::Point> features;
    int maxNum = 500;
    int blockSize = 3;
    cv::namedWindow(GetName());
    bool findFeatures = false;
    bool canTrack = false;
    cv::Size trWin(7, 7);
    int iterNum = 5;

    while (true)
    {
        capture >> frame;
        cv::cvtColor(frame, currentGray, CV_BGR2GRAY);

        if (findFeatures)
        {
            goodFeaturesToTrack(currentGray, features, maxNum, 0.01, 10, cv::Mat(), blockSize, 0, 0.04);
            findFeatures = false;
            canTrack = true;

            prevGray = currentGray.clone();
        }
        else if (canTrack)
        {
            // Tomasi-Kanade algorythm
            for (int t = 0; t < iterNum; t++)
            {
                for (size_t n = 0; n < features.size(); n++)
                {
                    cv::Mat C(2, 2, CV_32FC1, cv::Scalar(0)), g(2, 1, CV_32FC1, cv::Scalar(0)), d(2, 1, CV_32FC1, cv::Scalar(0));
                    int xLeft, xRight, yLow, yHigh;
                    float Ix, Iy, It;

                    xLeft = features[n].x - (trWin.width - 1) / 2;
                    xRight = features[n].x + (trWin.width - 1) / 2;
                    yLow = features[n].y - (trWin.height - 1) / 2;
                    yHigh = features[n].y + (trWin.height - 1) / 2;
                    if (xLeft < 0 || yLow < 0 || xRight >= prevGray.cols - 1 || yHigh >= prevGray.rows - 1)
                        continue;

                    for (int i = yLow; i < yHigh; i++)
                    {
                        for (int j = xLeft; j < xRight; j++)
                        {
                            Ix = static_cast<float>(prevGray.at<uchar>(i, j + 1)) - static_cast<float>(prevGray.at<uchar>(i, j));
                            Iy = static_cast<float>(prevGray.at<uchar>(i + 1, j)) - static_cast<float>(prevGray.at<uchar>(i, j));
                            It = static_cast<float>(currentGray.at<uchar>(i, j)) - static_cast<float>(prevGray.at<uchar>(i, j));

                            C.at<float>(0, 0) += Ix * Ix;
                            C.at<float>(0, 1) += Ix * Iy;
                            C.at<float>(1, 0) += Ix * Iy;
                            C.at<float>(1, 1) += Iy * Iy;

                            g.at<float>(0, 0) -= It * Ix;
                            g.at<float>(1, 0) -= It * Iy;
                        }
                    }

                    d = C.inv() * g;

                    features[n].x += static_cast<int>(d.at<float>(0, 0) + 0.5);
                    features[n].y += static_cast<int>(d.at<float>(1, 0) + 0.5);
                }
            }
        }

        for (size_t n = 0; n < features.size(); n++)
        {
            cv::circle(frame, features[n], 3, cv::Scalar(255, 255, 1), -1);
        }

        prevGray = currentGray.clone();

        cv::imshow(GetName(), frame);
        char c = (char)cv::waitKey(10);
        if (c == 27)
            return;
        switch (c)
        {
        case 'r':
            findFeatures = true;
            break;
        case 'c':
            features.clear();
            canTrack = false;
            break;
        }
    }
}

void ObjectTrackingTK::help()
{
    std::cout << "\nThis is a demo of Tomasi-Kanade algorythm\n" <<
        "\nHot keys: \n"
        "\tESC - quit the program\n"
        "\tr - auto-initialize tracking\n"
        "\tc - delete all the points\n" << std::endl;
}
