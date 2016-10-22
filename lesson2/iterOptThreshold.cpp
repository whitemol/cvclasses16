///@File: kmeans.cpp
///@Brief: Contains implementation of k-means algorithm for image segmentation.
///@Author: Alexander Kirilkin
///@Date: 05 October 2016

#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <limits>
#include <functional>
#include <chrono>
#include <ctime>

static void help()
{
	std::cout <<	"\nThis program demonstrates the Iterative Optimal Thresholding segmentation algorithm in OpenCV.\n"
					"Usage:\n"
					"./iterOptThreshold [image_name -- default is GrayPic.png]\n" << std::endl;
	std::cout <<	"Hot keys: \n"
					"\tESC - quit the program\n"
					"\tr - restore the original image\n"
					"\tw or SPACE - run iterative optimal thresholding segmentation algorithm\n"
					"\t\t(before running it, put start value of T, and delta)\n";
}

void iterOptThrHist(cv::Mat img, int T, int step)
{
	int histSize = 256; //from 0 to 255
	float range[] = { 0, 256 }; //the upper boundary is exclusive
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	cv::Mat hist;
	// Compute the histograms:
	cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
	// Draw the histograms for R, G and B
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));
	// Normalize the result to [ 0, histImage.rows ]
	cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			cv::Scalar(0, 0, 0), 2, 8, 0);
	}
	line(histImage, cv::Point(T * 2, hist_h), cv::Point(T * 2, 0), cv::Scalar(0, 200, 0), 2, 8, 0);
	cv::namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
	cv::imshow("calcHist Demo", histImage);
	std::string name = "iterOptThreshold_" + std::to_string(step) + "_hist.jpg";
	cv::imwrite(name, histImage);
	cv::waitKey(0);
}

cv::Mat img_in_iterOptThreshold, StepIm;
int step = 0;
std::chrono::time_point<std::chrono::system_clock> start, end;
double elapsed_time_iterOpt{};

int iterOptThreshold(int argc, char** argv)
{
	cv::CommandLineParser parser(argc, argv, "{help h | | }{ @input | GrayPic.png | }");
	if (parser.has("help"))
	{
		help();
		return 0;
	}
	std::string filename = parser.get<std::string>("@input");
	cv::Mat original_img_iterOptThreshold = cv::imread(filename, 0);
	//cvtColor(original_img_iterOptThreshold, imgGray, CV_BGR2GRAY);

	if (original_img_iterOptThreshold.empty())
	{
		std::cout << "Couldn't open image " << filename << ". Usage: iterOptThreshold <image_name>\n";
		return 0;
	}

	help();

	std::cout << "Input start threshold T\n";
	float Threshold;
	float diff = 256;
	float eps;
	std::cin >> Threshold;
	std::cout << "Input eps\n";
	std::cin >> eps;

	cv::namedWindow("image", 0);

	original_img_iterOptThreshold.copyTo(img_in_iterOptThreshold);
	cv::imshow("image", img_in_iterOptThreshold);

	for (;;)
	{
		int c = cv::waitKey(0);
		if ((char)c == 27)
			break;
		if ((char)c == 'r')
		{
			original_img_iterOptThreshold.copyTo(img_in_iterOptThreshold);
			imshow("image", img_in_iterOptThreshold);
		}

		if ((char)c == 'w' || (char)c == ' ')
		{
			float* m = new float[2];
			uint* counter = new uint[2];
			while (diff > eps)
			{
				start = std::chrono::system_clock::now();
				m[0] = 0;
				m[1] = 0;
				counter[0] = 0;
				counter[1] = 0;
				for (int y(img_in_iterOptThreshold.rows - 1); y >= 0; --y)
				{
					unsigned char *const scanLine(img_in_iterOptThreshold.ptr<unsigned char>(y));

					for (int x(img_in_iterOptThreshold.cols - 1); x >= 0; --x)
					{
						if (scanLine[x] <= Threshold)
						{
							m[0] += scanLine[x];
							counter[0]++;
						}
						else
						{
							m[1] += scanLine[x];
							counter[1]++;
						}
					}
				}
				if (counter[0] == 0)
				{
					counter[0] = 1;
					std::cout << "There is no values < Threshold: " << Threshold << std::endl;
				}
				diff = std::abs(Threshold - ((m[0] / counter[0]) + (m[1] / counter[1])) / 2);
				Threshold = ((m[0] / counter[0]) + (m[1] / counter[1])) / 2;

				end = std::chrono::system_clock::now();
				elapsed_time_iterOpt += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				
				std::cout << "m1 = " << m[0] / counter[0] << " | m2 = " << m[1] / counter[1] << " | T = " << Threshold 
					<< " | t_current = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
					<< " | t_collect = " << elapsed_time_iterOpt << std::endl;
				
				iterOptThrHist(img_in_iterOptThreshold, Threshold, step);
				img_in_iterOptThreshold.copyTo(StepIm);
				for (int y(StepIm.rows - 1); y >= 0; --y)
				{
					unsigned char *const scanLine(StepIm.ptr<unsigned char>(y));

					for (int x(StepIm.cols - 1); x >= 0; --x)
					{
						if (scanLine[x] <= Threshold)
						{
							scanLine[x] = 0;
						}
						else
						{
							scanLine[x] = 255;
						}
					}
				}
				std::string name = "iterOptThreshold_frame_" + std::to_string(step) + ".jpg";
				cv::imwrite(name, StepIm);
				cv::imshow("image", StepIm);
				step++;
			}
		}
	}
	return 0;
}