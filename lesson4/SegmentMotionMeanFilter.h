#pragma once

#include "SegmentMotionBase.h"

#include "opencv2\core\mat.hpp"

class SegmentMotionMeanFilter : public SegmentMotionBase
{
public:
  SegmentMotionMeanFilter()
  {
  }

  virtual std::string GetName() const override
  {
    return "SegmentMotionMeanFilter";
  }

protected:
  virtual cv::Mat process(cv::Mat& file_name) override;

  virtual void createGUI() override;

  void updateBackground(cv::Mat& currentFrame);

  struct Params
  {
    int threshold;
    int alpha;
    int num_mean;
    int counter;
  };

  Params m_params;
  cv::Mat Background;
};
