#include "SegmentMotionMeanFilter.h"

#include <iostream>

#include "opencv2\videoio.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\highgui.hpp"

///////////////////////////////////////////////////////////////////////////////
cv::Mat SegmentMotionMeanFilter::process(cv::Mat& currentFrame)
{
  cvtColor(currentFrame, currentFrame, CV_RGB2GRAY);
  updateBackground(currentFrame);

  cv::Mat result = abs(Background - currentFrame.clone());
  cv::threshold(result, result, m_params.threshold, 255, CV_THRESH_BINARY);

  return result;
}

///////////////////////////////////////////////////////////////////////////////
void SegmentMotionMeanFilter::createGUI()
{
  const std::string windowName = GetName();
  cv::namedWindow(windowName);

  m_params.alpha = 10;
  m_params.threshold = 10;
  m_params.num_mean = 30*3;
  m_params.counter = 0;

  cv::createTrackbar("Threshold", windowName, &m_params.threshold, 255);
  cv::createTrackbar("Alpha", windowName, &m_params.alpha, 100);
}

///////////////////////////////////////////////////////////////////////////////
void SegmentMotionMeanFilter::updateBackground(cv::Mat& currentFrame)
{
  float scaled_alpha = m_params.alpha * 0.01f;

  if (Background.empty()) {
    Background = currentFrame.clone();
    m_params.counter++;
    return;
  }

  cv::Mat_<float> floatBackground(Background);
  cv::Mat_<float> floatCurrentFrame(currentFrame);

  if (m_params.counter >= m_params.num_mean) {
    floatBackground = (1 - scaled_alpha) * floatBackground + scaled_alpha * floatCurrentFrame;
  } else {
    floatBackground = floatBackground * m_params.counter + floatCurrentFrame;
    m_params.counter++;
    floatBackground /= m_params.counter;
  }

  floatBackground.convertTo(Background, CV_8U);
}
