///@File: main.cpp
///@Brief: Contains implementation of entry point of the application
///@Author: Roman Golovanov
///@Date: 08 September 2015

#include "stdafx.h"

#include "LaplacianEdgeDetector.h"

#define PARSE_CODEC(codec)                            \
	CV_FOURCC(codec[0], codec[1], codec[2], codec[3]) \

const cv::String keys =
    "{help h usage ? |      | Hepl info              }"
    "{@filename      |      | Output video file.     }"
    "{codec          | MP43 | Codec (4 letter).      }"
    "{fps            | 30.0 | FPS for output video.  }"
    "{width          | 640  | Width of output video. }"
    "{height         | 480  | Height of output video.}"
    ;

///@brief Entry point
int main(int argc, char** argv )
{
	cv::CommandLineParser parser(argc, argv, keys);


	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}

	int height = parser.get<int>("height");
	int width =  parser.get<int>("width");

	double fps = parser.get<double>("fps");

	std::string codec = parser.get<std::string>("codec");
	std::string filename = parser.get<std::string>(0);

	if (height < 20 || height > 1080) {
		std::cerr << "Height should be in the range from 20 to 1080" << std::endl;
		return -1;
	}

	if (width < 20 || width > 1920) {
		std::cerr << "Width should be in the range from 20 to 1920" << std::endl;
		return -2;
	}

	if (fps < 1. || fps > 120.) {
		std::cerr << "FPS should be in the range from 1.0 to 120.0" << std::endl;
		return -3;
	}

	if (codec.size() != 4) {
		std::cerr << "Codec is a string which consist of 4 characters." << std::endl;
		return -4;
	}


	LaplacianEdgeDetector detector;
	if (!filename.empty())
		if (detector.InitFrameWriter(filename, PARSE_CODEC(codec), fps, cv::Size(width, height)))
			return -5;

	detector.Show();
	return 0;
}
