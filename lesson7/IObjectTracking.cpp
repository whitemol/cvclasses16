#include "stdafx.h"
#include "IObjectTracking.h"


void ObjectTracking::mark_stay_points(cv::Mat &image, const std::vector<cv::Point2f>& points, 
									  const std::vector<size_t>& stay_count,
									  const std::vector<bool>& mov_any)
{
	std::vector<cv::Rect2i> all_rectangles;
	std::vector<cv::Rect2i> merged_rectangles;

	for (size_t i = 0; i < stay_count.size(); ++i)
		if (stay_count[i] > 30 && mov_any[i])
			all_rectangles.push_back(cv::Rect2i(cv::Point(points[i].x - 80 / 2,
			                                              points[i].y - 80 / 2),       
														  cv::Size(80, 80)));

	std::vector<int> mask(all_rectangles.size(), 1);

	for (size_t i = 0; i < all_rectangles.size(); ++i) {

		if (!mask[i])
			continue;

		cv::Rect2i curr_rectangle = all_rectangles[i];

		for (size_t j = i + 1; j < all_rectangles.size(); ++j) {
			if (mask[j]) {
				if ((curr_rectangle & all_rectangles[j]).area()) {
					curr_rectangle = curr_rectangle | all_rectangles[j];
					mask[j] = 0;
				}
			}
		}

		merged_rectangles.push_back(curr_rectangle);
	}

	for (auto &rect : merged_rectangles)
		cv::rectangle(image, rect, {0, 0, 255});
}


void ObjectTracking::Run(cv::VideoCapture & capture, cv::Mat& background,
						 cv::VideoWriter& writer)
{
	const int MAX_COUNT = 500;
	bool needToInit = true;

	cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
	cv::Size subPixWinSize(10, 10), winSize(31, 31);

	cv::namedWindow(GetName(), 1);

	cv::Mat gray, prevGray, image, frame;
	std::vector<cv::Point2f> points[2];
	std::vector<bool> mov_any;
	std::vector<size_t> stay_count[2];

	int win_Size = 3;
	cv::createTrackbar("WinSize", GetName(), &win_Size, 10);

	while (1) {
		capture >> frame;
		if (frame.empty())
			break;

		frame.copyTo(image);
		cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

		if (needToInit) {
			std::vector<cv::Point2f> bg_points;

			cv::goodFeaturesToTrack(background, bg_points, MAX_COUNT, 0.001, 10, cv::Mat(), 3, 0, 0.04);
			cv::cornerSubPix(background, bg_points, subPixWinSize, cv::Size(-1, -1), termcrit);

			std::vector<cv::Point2f> frame_points;

			cv::goodFeaturesToTrack(gray, frame_points, MAX_COUNT, 0.001, 10, cv::Mat(), 3, 0, 0.04);
			cv::cornerSubPix(gray, frame_points, subPixWinSize, cv::Size(-1, -1), termcrit);

			for (auto &cp : frame_points) {
				bool add_flag = true;
				for (auto &bp : bg_points) {
					if (norm(cp - bp) < 3.0 ) {
						add_flag = false;
						break;
					}
				}

				if (add_flag)
					points[1].push_back(cp);
			}

			stay_count[1].resize(points[1].size(), 0);
			mov_any.resize(points[1].size(), 0);
		}
		else if (!points[0].empty()) {
			std::vector<uchar> status;
			std::vector<float> err;

			if (prevGray.empty())
				gray.copyTo(prevGray);

			cv::calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
								     5, termcrit, 0, 0.001);

			stay_count[1].clear();
			size_t i, k;
			for (i = k = 0; i < points[1].size(); i++) {

				if (!status[i])
					continue;

				if (cv::norm(points[0][i] - points[1][i]) < 2.0) {
					stay_count[1].push_back(stay_count[0][i] + 1);
				} else {
					stay_count[1].push_back(0);
					mov_any[k] = 1 | mov_any[i];
				}

				if (mov_any[i])
					//cv::circle(image, points[1][i], 3, cv::Scalar(0, 0, 255), -1, 8);
				//else
					cv::circle(image, points[1][i], 3, cv::Scalar(0, 255, 0), -1, 8);

				points[1][k] = points[1][i];
				//circle(image, points[1][i], 3, cv::Scalar(0, 255, 0), -1, 8);

				k++;
			}
			mov_any.resize(k);
			points[1].resize(k);
		}

		mark_stay_points(image, points[1], stay_count[1], mov_any);

		needToInit = false;
		writer << image;
		cv::imshow(GetName(), image);

		char c = (char) cv::waitKey(10);
		if (c == 27)
			return;
		switch (c)
		{
			case 'r':
				needToInit = true;
				points[0].clear();
				points[1].clear();
				break;
			case 'c':
				points[0].clear();
				points[1].clear();
				break;
		}

		std::swap(stay_count[1], stay_count[0]);
		std::swap(points[1], points[0]);
		cv::swap(prevGray, gray);
	}
}
