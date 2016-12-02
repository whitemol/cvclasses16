///@File: ObjectTrackingSTK.cpp
///@Brief: implementation of ObjectTrackingSTK class
///@Author: Sidorov Stepan and Kuksova Svetlana
///@Date: 20.12.2015

#include "stdafx.h"
#include "ObjectTrackingSTK.h"

void ObjectTrackingSTK::Run(cv::VideoCapture & capture)
{
    help();

    cv::Mat frame, currentGray, prevGray;
    std::vector<cv::Point> features;
    int maxNum = 500;
    int blockSize = 3;
    cv::namedWindow(GetName());
    bool findFeatures = false;
    bool canTrack = false;
    cv::Size trWin(3, 3);
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
            // Shi-Tomasi-Kanade algorythm
            for (int t = 0; t < iterNum; t++)
            {
                for (size_t n = 0; n < features.size(); n++)
                {
                    cv::Mat T(6, 6, CV_32FC1, cv::Scalar(0)), T1(6, 6, CV_32FC1, cv::Scalar(0)), a(6, 1, CV_32FC1, cv::Scalar(0)), z(6, 1, CV_32FC1, cv::Scalar(0));
                    cv::Mat U(4, 4, CV_32FC1, cv::Scalar(0)), V(4, 2, CV_32FC1, cv::Scalar(0)), G(2, 2, CV_32FC1, cv::Scalar(0));
                    int xLeft, xRight, yLow, yHigh;
                    float Iu, Iv, It;

                    xLeft = features[n].x - (trWin.width - 1) / 2;
                    xRight = features[n].x + (trWin.width - 1) / 2;
                    yLow = features[n].y - (trWin.height - 1) / 2;
                    yHigh = features[n].y + (trWin.height - 1) / 2;
                    if (xLeft < 0 || yLow < 0 || xRight >= prevGray.cols - 1 || yHigh >= prevGray.rows - 1)
                        continue;

                    for (int v = yLow; v < yHigh; v++)
                    {
                        for (int u = xLeft; u < xRight; u++)
                        {
                            Iu = static_cast<float>(prevGray.at<uchar>(v, u + 1)) - static_cast<float>(prevGray.at<uchar>(v, u));
                            Iv = static_cast<float>(prevGray.at<uchar>(v + 1, u)) - static_cast<float>(prevGray.at<uchar>(v, u));
                            It = static_cast<float>(currentGray.at<uchar>(v, u)) - static_cast<float>(prevGray.at<uchar>(v, u));

                            U.at<float>(0, 0) += u * u * Iu * Iu;
                            U.at<float>(0, 1) += u * u * Iu * Iv;
                            U.at<float>(0, 2) += u * v * Iu * Iu;
                            U.at<float>(0, 3) += u * v * Iu * Iv;
                            U.at<float>(1, 0) += u * u * Iu * Iv;
                            U.at<float>(1, 1) += u * u * Iv * Iv;
                            U.at<float>(1, 2) += u * v * Iu * Iv;
                            U.at<float>(1, 3) += u * v * Iv * Iv;
                            U.at<float>(2, 0) += u * v * Iu * Iu;
                            U.at<float>(2, 1) += u * v * Iu * Iv;
                            U.at<float>(2, 2) += v * v * Iu * Iu;
                            U.at<float>(2, 3) += v * v * Iu * Iv;
                            U.at<float>(3, 0) += u * v * Iu * Iv;
                            U.at<float>(3, 1) += u * v * Iv * Iv;
                            U.at<float>(3, 2) += v * v * Iu * Iv;
                            U.at<float>(3, 3) += v * v * Iv * Iv;

                            V.at<float>(0, 0) += u * Iu * Iu;
                            V.at<float>(0, 1) += u * Iu * Iv;
                            V.at<float>(1, 0) += u * Iu * Iv;
                            V.at<float>(1, 1) += u * Iv * Iv;
                            V.at<float>(2, 0) += v * Iu * Iu;
                            V.at<float>(2, 1) += v * Iu * Iv;
                            V.at<float>(3, 0) += v * Iu * Iv;
                            V.at<float>(3, 1) += v * Iv * Iv;

                            G.at<float>(0, 0) += Iu * Iu;
                            G.at<float>(0, 1) += Iu * Iv;
                            G.at<float>(1, 0) += Iu * Iv;
                            G.at<float>(1, 1) += Iv * Iv;

                            a.at<float>(0, 0) -= It * u * Iu;
                            a.at<float>(1, 0) -= It * u * Iv;
                            a.at<float>(2, 0) -= It * v * Iu;
                            a.at<float>(3, 0) -= It * v * Iv;
                            a.at<float>(4, 0) -= It * Iu;
                            a.at<float>(5, 0) -= It * Iv;
                        }
                    }

                    cv::Mat temp1, temp2;
                    cv::vconcat(U, V.t(), temp1);
                    cv::vconcat(V, G, temp2);
                    cv::hconcat(temp1, temp2, T);

                    z = T.inv() * a;

                    features[n].x = ::lroundf(features[n].x * (1 + z.at<float>(0, 0)) + features[n].y * z.at<float>(1, 0) + z.at<float>(4, 1));
                    features[n].y = ::lroundf(features[n].x * z.at<float>(2, 0) + features[n].y * (1 + z.at<float>(3, 0)) + z.at<float>(5, 1));
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

void ObjectTrackingSTK::help()
{
    std::cout << "\nThis is a demo of Shi-Tomasi-Kanade algorythm\n" <<
        "\nHot keys: \n"
        "\tESC - quit the program\n"
        "\tr - auto-initialize tracking\n"
        "\tc - delete all the points\n" << std::endl;
}
