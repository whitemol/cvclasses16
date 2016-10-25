///@File: main.cpp
///@Brief: Contains implementation of entry point of the application
///@Author: Roman Golovanov
///@Date: 08 September 2015


#include "stdafx.h"

#include <iostream>
#include <stdio.h>
#include <functional>
#include <fstream>

#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"


enum FILTER_TYPE { PREWITT, DOG };
float prewitt_mask_fx[] = { -1, 0, 1,
							-1, 0, 1,
							-1, 0, 1 };
float prewitt_mask_fy[] = { -1, -1, -1,
							 0,  0,  0,
							 1,  1,  1 };
float dog_mask[] = { -0.0151, 0.0026, 0.0184, 0.0243, 0.0184, 0.0026, -0.0151,
					0.0026, 0.0300, 0.0380, 0.0310, 0.0380, 0.0300, 0.0026,
					0.0184, 0.0380, -0.0275, -0.1020, -0.0275, 0.0380, 0.0184,
					0.0243, 0.0310, -0.1020, -0.2355, -0.1020, 0.0310, 0.0243,
					0.0184, 0.0380, -0.0275, -0.1020, -0.0275, 0.0380, 0.0184,
					0.0026, 0.0300, 0.0380, 0.0310, 0.0380, 0.0300, 0.0026,
					-0.0151, 0.0026, 0.0184, 0.0243, 0.0184, 0.0026, -0.0151 };


void apply1stDerivativeAlgo(cv::Mat& i_img, cv::Mat& o_edges, int i_threshold, FILTER_TYPE i_type)
{
	cv::Mat dX, dY, dMag;

	switch (i_type)
	{
	case PREWITT:
		cv::filter2D(i_img, dX, CV_32F, cv::Mat(3, 3, CV_32F, prewitt_mask_fx));
		cv::filter2D(i_img, dY, CV_32F, cv::Mat(3, 3, CV_32F, prewitt_mask_fy));
		cv::magnitude(dX, dY, dMag);
		break;

	default:
		std::cerr << "unsupported filter type" << std::endl;
	}

	double maxVal = 0;
	cv::minMaxLoc(dMag, nullptr, &maxVal);
	const double th = 0.01 * i_threshold * maxVal;
	cv::threshold(dMag, o_edges, th, 255, CV_THRESH_BINARY);
}


void apply2ndDerivativeAlgo(cv::Mat& i_img, cv::Mat& o_edges, int i_threshold, FILTER_TYPE i_type)
{
	cv::Mat deriv2;

	switch (i_type)
	{
	case DOG:
		cv::filter2D(i_img, deriv2, CV_32F, cv::Mat(7, 7, CV_32F, dog_mask));
		break;

	default:
		std::cerr << "unsupported filter type" << std::endl;
	}

	double minVal = 0, maxVal = 0;
	cv::minMaxLoc(deriv2, &minVal, &maxVal);
	const double th = 0.01 * i_threshold * (std::max)(maxVal, std::abs(minVal));
	o_edges = cv::Mat::zeros(deriv2.size(), CV_8U);
	for (int x = 1; x < deriv2.rows - 1; ++x)
	{
		for (int y = 1; y < deriv2.cols - 1; ++y)
		{
			const cv::Mat nghb = deriv2.rowRange({ x - 1, x + 2 }).colRange({ y - 1, y + 2 });

			if (nghb.at<float>(0, 0) * nghb.at<float>(2, 2) < 0 && std::abs(nghb.at<float>(0, 0) - nghb.at<float>(2, 2)) > th ||
				nghb.at<float>(1, 0) * nghb.at<float>(1, 2) < 0 && std::abs(nghb.at<float>(1, 0) - nghb.at<float>(1, 2)) > th ||
				nghb.at<float>(2, 0) * nghb.at<float>(0, 2) < 0 && std::abs(nghb.at<float>(2, 0) - nghb.at<float>(0, 2)) > th ||
				nghb.at<float>(2, 1) * nghb.at<float>(0, 1) < 0 && std::abs(nghb.at<float>(2, 1) - nghb.at<float>(0, 1)) > th)
			{
				o_edges.at<uchar>(x, y) = 255;
			}
		}
	}
}


void quality_metric(cv::Mat& i_edges, cv::Mat& i_manual, double* o_precision, double* o_recall)
{
	int fp = 0, fn = 0, tp = 0, tn = 0;
	for (int x = 0; x < i_edges.rows; x++)
	{
		for (int y = 0; y < i_edges.cols; y++)
		{
			if (i_manual.at<uchar>(x, y) == 255)
			{
				if (i_edges.at<uchar>(x, y) == 255)
					tp++;
				else
					fp++;
			}
			else
			{
				if (i_edges.at<uchar>(x, y) == 255)
					fn++;
				else
					tn++;
			}

		}
	}
	*o_precision = tp / double(tp + fp);
	*o_recall = tp / double(tp + fn);
}


void processFrame(cv::Mat& i_image, cv::Mat& i_manual, int i_threshold, std::ofstream& i_output)
{
	cv::Mat edges;
	const int bufSize = 100;
	char buffer[bufSize];
	double precision = 0, recall = 0;

	apply1stDerivativeAlgo(i_image, edges, i_threshold, PREWITT);
	edges.convertTo(edges, CV_8U);
	quality_metric(edges, i_manual, &precision, &recall);
	i_output << i_threshold << " " << precision << " " << recall << std::endl;

	sprintf_s(buffer, bufSize, "prewitt_%d.bmp", i_threshold);
	cv::imwrite(buffer, edges);

	apply2ndDerivativeAlgo(i_image, edges, i_threshold, DOG);
	quality_metric(edges, i_manual, &precision, &recall);
	i_output << i_threshold << " " << precision << " " << recall << std::endl;

	sprintf_s(buffer, bufSize, "dog_%d.bmp", i_threshold);
	cv::imwrite(buffer, edges);
}


int main(int argc, char** argv)
{
	cv::CommandLineParser parser(argc, argv, "{ help h   |              | }"
											 "{ image    | original.bmp | }"
											 "{ edges    | edges.bmp    | }");
	if (parser.has("help"))
	{
		std::cout << "Usage: lesson3 <image_name> <edge_name>" << std::endl;
		return 0;
	}

	const auto& imageName = parser.get<std::string>("image");
	const auto& edgesName = parser.get<std::string>("edges");

	cv::Mat image = cv::imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);
	if (image.empty())
	{
		std::cerr << "Couldn'g open image " << imageName << ". Usage: lesson3 <image_name> <edge_name>" << std::endl;
		return -1;
	}

	cv::Mat edges = cv::imread(edgesName, CV_LOAD_IMAGE_GRAYSCALE);
	if (edges.empty())
	{
		std::cerr << "Couldn'g open image " << edges << ". Usage: lesson3 <image_name> <edge_name>" << std::endl;
		return -1;
	}

	std::ofstream output("out.txt");
	for (int th = 0; th < 100; th += 20)
		processFrame(image, edges, th, output);
	output.close();

	return 0;
}
