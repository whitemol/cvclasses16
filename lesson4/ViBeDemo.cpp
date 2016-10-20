
#include "ViBe.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

using namespace cv;

void TrackbarFunction(int pos, void* userdata)
{
    double* learning_rate = (double*)userdata;
    *learning_rate = pos == 0 ? 0 : 1.0 / pos;
    return;
}

void ViBeDemo()
{    
    Mat frame;
    VideoCapture video(0);
    namedWindow("Original");
    for (int i = 0; i < 50; ++i)
    {
        video >> frame;
        imshow("Original", frame);
        waitKey(30);
    }

    
    Mat bgmask;
    Mat background;
    uchar c = 0;
    int trackbar_value = 10;
    ViBe vibe(20, 20, 2, 10);
    namedWindow("ViBe");
    double learning_rate = 0.0;
    createTrackbar("Learning_Rate", "ViBe", &trackbar_value, 50, TrackbarFunction, &learning_rate);
    namedWindow("BackGround");
    

    while (true)
    {
        video >> frame;
        imshow("Original", frame);

        vibe.apply(frame, bgmask, learning_rate);
        imshow("ViBe", bgmask);

        vibe.getBackgroundImage(background);
        imshow("BackGround", background);

        // Выход из цикла при нажатии Esc.
        c = waitKey(33);
        if (c == 27)
            break;
    }
    
    destroyAllWindows();
    return;
}