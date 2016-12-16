#include "stdafx.h"

#pragma once

class ObjectTracking
{
public:

	void Run(cv::VideoCapture& capture, cv::Mat& background,
			 cv::VideoWriter& writer);

	void mark_stay_points(cv::Mat& image, const std::vector<cv::Point2f>& points,
						  const std::vector<size_t>& stay_count,
						  const std::vector<bool>& mov_any);

	std::string GetName() const
	{
		return "LK demo.";
	}
};
