#include "demosift.h"

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include <iostream>
#include <limits>
#include <vector>
#include <functional>
#include <chrono>
#include <ctime>

size_t num_octaves;
size_t num_images;
double init_sigma;
double k;

static void help()
{
	std::cout << "\n This program demonstrates work of SIFT algorithm."
	          << std::endl;
}

const cv::String keys =
    "{help h usage ? |                    | Hepl info                         }"
    "{@ref_image     | ref_image.jpg      | Reference image.                  }"
    "{@test_image    | test_image.jpg     | Test image.                       }"
    "{num_octaves    | 4                  | Number of octaves in the pyramid. }"
    "{num_images     | 5                  | Number images in octave           }"
    ;


std::tuple<uint8_t, uint8_t> extr_win(const cv::Mat &mat1,
                                      const cv::Mat &mat2,
                                      const cv::Mat &mat3,
                                      const size_t &row, const size_t &col)
{
	uint8_t max_val = std::numeric_limits<uint_fast8_t>::min();
	uint8_t min_val = std::numeric_limits<uint_fast8_t>::max();

	for (size_t i = row - 1; i <= row + 1; ++i)
		for (size_t j = col - 1; j <= col + 1; ++j) {
			if (i != row && j != col) {
				max_val = std::max(max_val, mat1.at<uint8_t>(i, j));
				max_val = std::max(max_val, mat2.at<uint8_t>(i, j));
				max_val = std::max(max_val, mat3.at<uint8_t>(i, j));

				min_val = std::min(min_val, mat1.at<uint8_t>(i, j));
				min_val = std::min(min_val, mat2.at<uint8_t>(i, j));
				min_val = std::min(min_val, mat3.at<uint8_t>(i, j));
			}
		}

	max_val = std::max(max_val, mat1.at<uint8_t>(row, col));
	max_val = std::max(max_val, mat3.at<uint8_t>(row, col));

	min_val = std::min(min_val, mat1.at<uint8_t>(row, col));
	min_val = std::min(min_val, mat3.at<uint8_t>(row, col));

	return std::tuple<uint8_t, uint8_t>(max_val, min_val);
}

void find_points(const cv::Mat &bottom_mat,
           const cv::Mat &central_mat,
           const cv::Mat &top_mat,
           std::vector<cv::Point2d> &points)
{
	size_t rows = bottom_mat.rows;
	size_t cols = bottom_mat.cols;

	for (size_t i = 1; i < rows - 1; ++i) {
		for (size_t j = 1; j < cols - 1; ++j) {
			uint8_t value = central_mat.at<uint8_t>(i, j);

			auto maxmin = extr_win(bottom_mat,
								   central_mat,
								   top_mat,
								   i, j);

			if (value > std::get<0>(maxmin) || value < std::get<1>(maxmin))
				points.push_back(cv::Point2d(i, j));

		}
	}
}


bool load_image(cv::Mat &image, const std::string &image_path)
{
	image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);

	if (image.empty()) {
		std::cerr << "Incorrect image path: " << image_path << std::endl;
		return false;
	}

	return true;
}


void get_dog_piramid(const cv::Mat& image,
                     std::vector<std::vector<cv::Mat>> &dof_piramid)
{
	cv::Size image_size = image.size();

	std::vector<std::vector<cv::Mat>> blurred_piramid(num_octaves);

	for (size_t oct = 0; oct < num_octaves; ++oct) {
		double sigma = init_sigma;
		cv::Mat resized_image;
		cv::resize(image, resized_image, cv::Size(image_size.width / std::pow(2, oct),
		                                          image_size.height / std::pow(2, oct)));
		for (size_t i = 0; i < num_images; ++i) {
			cv::Mat result;
			cv::GaussianBlur(resized_image, result, cv::Size(21, 21), sigma);
			blurred_piramid[oct].push_back(result);
			sigma *= k;
		}
	}

	dof_piramid.resize(num_octaves);
	for (size_t i = 0; i < blurred_piramid.size(); ++i) {
		for (size_t j = 1; j < blurred_piramid[i].size(); ++j) {
			cv::Mat diff = blurred_piramid[i][j] - blurred_piramid[i][j - 1];
			dof_piramid[i].push_back(diff);
		}
	}

}


void get_points(std::vector<std::vector<cv::Point2d>> &points,
                const std::vector<std::vector<cv::Mat>> &dof_piramid)
{
	points.resize(num_octaves);

	for (size_t i = 0; i < dof_piramid.size(); ++i) {
		for (size_t j = 1; j < dof_piramid[i].size() - 1; ++j) {
				find_points(dof_piramid[i][j-1], dof_piramid[i][j], dof_piramid[i][j+1], points[i]);
		}
	}
}


void demoSIFT(int argc, char** argv) {
	help();

	// Parse parameters

	cv::CommandLineParser parser(argc, argv, keys);

	if (parser.has("help")) {
		std::cerr << "Incorrect parameters" << std::endl;
		parser.printMessage();
		return;
	}

	const std::string ref_file = parser.get<std::string>(0);
	const std::string test_file = parser.get<std::string>(1);

	num_octaves = parser.get<size_t>("num_octaves");
	num_images = parser.get<size_t>("num_images");

	init_sigma = 1.6;
	k = std::sqrt(2);

	// Load images
	cv::Mat ref_image;
	cv::Mat test_image;

	load_image(ref_image, ref_file);
	load_image(test_image, test_file);

	// RGB2GRAY
	cv::Mat g_ref_image;
	cv::Mat g_test_image;

	cv::cvtColor(ref_image, g_ref_image, cv::COLOR_RGB2GRAY);
	cv::cvtColor(test_image, g_test_image, cv::COLOR_RGB2GRAY);

	// DoG piramids
	std::vector<std::vector<cv::Mat>> ref_dog_piramid;
	std::vector<std::vector<cv::Mat>> test_dog_piramid;

	get_dog_piramid(g_ref_image, ref_dog_piramid);
	get_dog_piramid(g_test_image, test_dog_piramid);

	// Find points
	std::vector<std::vector<cv::Point2d>> ref_points;
	std::vector<std::vector<cv::Point2d>> test_points;

	get_points(ref_points, ref_dog_piramid);
	get_points(test_points, test_dog_piramid);

	// TODO: Point filtration

	// TODO: Compare images

	// TODO: Show result

	cv::imshow("Reference image", ref_image);
	cv::imshow("Test image", test_image);
	cv::imshow("Test result", ref_dog_piramid[0][0]);

	cv::waitKey(0);

	cv::destroyAllWindows();

}
