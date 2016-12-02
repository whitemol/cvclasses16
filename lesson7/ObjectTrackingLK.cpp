///@File: ObjectTrackingLK.cpp
///@Brief: implementation of ObjectTrackingLK class
///@Author: Sidorov Stepan
///@Date: 07.12.2015

#include "stdafx.h"
#include "ObjectTrackingLK.h"

cv::Point2f point;
bool addRemovePt = false;

void ObjectTrackingLK::help()
{
    std::cout << "\nThis is a demo of Lukas-Kanade optical flow lkdemo()\n" <<
        "\nHot keys: \n"
        "\tESC - quit the program\n"
        "\tr - auto-initialize tracking\n"
        "\tc - delete all the points\n"
        "\tn - switch the \"night\" mode on/off\n"
        "To add/remove a feature point click it\n" << std::endl;
}

void ObjectTrackingLK::Run(cv::VideoCapture & capture)
{
    help();

    const int MAX_COUNT = 500;
    bool needToInit = false;
    bool nightMode = false;

    cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
    cv::Size subPixWinSize(10, 10), winSize(31, 31);

    cv::namedWindow(GetName(), 1);
    cv::setMouseCallback(GetName(), onMouse, 0);

    cv::Mat gray, prevGray, image, frame;
    std::vector<cv::Point2f> points[2];

    int win_Size = 3;
    cv::createTrackbar("WinSize", GetName(), &win_Size, 10);

    for (;;)
    {
        capture >> frame;
        if (frame.empty())
            break;

        frame.copyTo(image);
        cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        if (nightMode)
            image = cv::Scalar::all(0);

        if (needToInit)
        {
            // automatic initialization
            goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, cv::Mat(), win_Size, 0, 0.04);
            cornerSubPix(gray, points[1], subPixWinSize, cv::Size(-1, -1), termcrit);
            addRemovePt = false;
        }
        else if (!points[0].empty())
        {
            std::vector<uchar> status;
            std::vector<float> err;
            if (prevGray.empty())
                gray.copyTo(prevGray);
            calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                5, termcrit, 0, 0.001);
            size_t i, k;
            for (i = k = 0; i < points[1].size(); i++)
            {
                if (addRemovePt)
                {
                    if (cv::norm(point - points[1][i]) <= 5)
                    {
                        addRemovePt = false;
                        continue;
                    }
                }

                if (!status[i])
                    continue;

                points[1][k++] = points[1][i];
                circle(image, points[1][i], 3, cv::Scalar(0, 255, 0), -1, 8);
            }
            points[1].resize(k);
        }

        if (addRemovePt && points[1].size() < (size_t)MAX_COUNT)
        {
            std::vector<cv::Point2f> tmp;
            tmp.push_back(point);
            cornerSubPix(gray, tmp, winSize, cv::Size(-1, -1), termcrit);
            points[1].push_back(tmp[0]);
            addRemovePt = false;
        }

        needToInit = false;
        imshow(GetName(), image);

        char c = (char)cv::waitKey(10);
        if (c == 27)
            return;
        switch (c)
        {
        case 'r':
            needToInit = true;
            break;
        case 'c':
            points[0].clear();
            points[1].clear();
            break;
        case 'n':
            nightMode = !nightMode;
            break;
        }

        std::swap(points[1], points[0]);
        cv::swap(prevGray, gray);
    }
}

void ObjectTrackingLK::onMouse(int event, int x, int y, int, void *)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        point = cv::Point2f((float)x, (float)y);
        addRemovePt = true;
    }
}
