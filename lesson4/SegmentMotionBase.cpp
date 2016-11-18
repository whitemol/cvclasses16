#include "SegmentMotionBase.h"

#include <iostream>

#include "opencv2\video\video.hpp"
#include "opencv2\highgui\highgui.hpp"

#include "SegmentMotionMeanFilter.h"

///////////////////////////////////////////////////////////////////////////////
void SegmentMotionBase::Run(const std::string &video_file)
{
  cv::VideoCapture capture;
  cv::VideoWriter outputVideo;

  try {
    capture.open(video_file);
    outputVideo.open(std::string("result.avi"), CV_FOURCC('M', 'J', 'P', 'G'), 30.0, { 800, 548 });
  }
  catch (...) {
    std::cerr << "Incorrect parameters for frameWriter." << std::endl;
    exit(-1);
  }


  if (!capture.isOpened()) {
    std::cerr << "Can not open the camera !" << std::endl;
    exit(-1);
  }

  createGUI();

  cv::Mat frame;

  while (true) {
    capture >> frame;

    if (frame.empty())
      break;

    m_foreground = process(frame);
    cv::imshow(GetName(), m_foreground);

    cv::Mat buf(frame.rows, frame.cols, CV_8UC3);

    cv::cvtColor(m_foreground, buf, cv::COLOR_GRAY2BGR);

    outputVideo.write(buf);

    if (cv::waitKey(1) >= 0)
      break;
  }
}

///////////////////////////////////////////////////////////////////////////////
SegmentMotionBase* SegmentMotionBase::CreateAlgorithm(std::string& algorithmName)
{
    if (algorithmName == "MM") {
      return new SegmentMotionMeanFilter();
    } else {
      return nullptr;
    }
}
