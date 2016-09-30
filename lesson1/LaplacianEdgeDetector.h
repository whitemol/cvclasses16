///@File: LaplacianEdgeDetector.h
///@Brief: Contains definition of LaplacianEdgeDetector class
///@Author: Roman Golovanov
///@Date: 08 September 2015

#pragma once

#include "opencv2/opencv.hpp"

///@class LaplacianEdgeDetector
///@brief Demonstrates the laplacian of gaussian edge detector
class LaplacianEdgeDetector
{
public:
	///@brief Launch demonstration for passed image
	void Show();

	///@brief Initialize writing file
	int InitFrameWriter(const std::string& file_name, const int& video_codec,
	                    const double& video_fps, const cv::Size& frame_size);

	///@brief Returns the string with full name of this detector
	static cv::String ReplyName()
	{
		return "Laplacian Edge Detector";
	}

private:
	///@brief Contains main configurable parametrs for demo
	struct AlgoContext
	{
		int gaussianKernelSize	= 3;	///< size of gaussian kernel NxN
		int gaussianSigma		= 0;	///< sigma of gaussian

		int laplacianKernelSize	= 3;	///< size of laplacian kernel NxN
		int laplacianScale		= 1;	///< scale of laplacian
		int laplacianDelta		= 15;	///< delta of laplacian

		cv::VideoCapture* videoCapture = nullptr;	///< image capturer
	};

	///@brief applies algorithm according to the passed data
	void processFrame();

	///@brief Write frame to the file
	void writeFrame(const cv::Mat& frame);

	///@brief Mouse handler
	static void onMouse(int event, int x, int y, int flags, void* param);

private:

	///@brief current parameters
	AlgoContext m_context;

	static const cv::Point sc_invalidMousePosition;
	std::pair<cv::Point, cv::Point> m_mouseLine;

	cv::VideoWriter frameWriter;
	cv::Size frameSize;
};
