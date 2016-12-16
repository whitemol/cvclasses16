#include "stdafx.h"
#include "IObjectTracking.h"

int main(int argc, char** argv)
{
	cv::CommandLineParser parser(argc, argv, 
        "{@input_video            | dd73_l7.avi      | }"
		"{@input_background_video | dd73_l7_back.avi | }"
		"{@output_video           | out.avi          | }");

	std::string input_video = parser.get<std::string>("@input_video");
	std::string input_background = parser.get<std::string>("@input_background_video");
	std::string output_video = parser.get<std::string>("@output_video");

	cv::VideoCapture capture;
	cv::VideoWriter writer;

	capture.open(input_background);
	if (!capture.isOpened()) {
		std::cerr << "Capture is not open." << std::endl;
		return -1;
	}
	cv::Mat background;// = cv::imread("dd73.jpg", cv::IMREAD_GRAYSCALE);
	capture >> background;
	cvtColor(background, background, cv::COLOR_BGR2GRAY);

	capture.open(input_video);
	if (!capture.isOpened()) {
		std::cerr << "Capture is not open." << std::endl;
		return -1;
	}

	writer.open(output_video, CV_FOURCC('M', 'J', 'P', 'G'),
				capture.get(CV_CAP_PROP_FPS), background.size());
	if (!writer.isOpened()) {
		std::cerr << "Writer is not open." << std::endl;
		return -1;
	}

	ObjectTracking obj;
	obj.Run(capture, background, writer);

	cv::destroyAllWindows();

    return 0;
}
