///@File: kmeans.cpp
///@Brief: Contains implementation of k-means algorithm for image segmentation.
///@Author: Alexander Tselousov
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

#define CLS_IMG "Choice clusters"
#define MSK_IMF(i) "Rusult after iteration " + std::to_string(i)
#define MAX_CLUSTERS 100
#define MAX_ITR 100
#define SQR(x) (x) * (x)

static void help()
{
	std::cout << "\nThis program demonstrates kmeans clustering.\n"
	             "It generates an image with random points, then assigns a random number of cluster\n"
	             "centers and uses kmeans to move those cluster centers to their representitive location\n"
	             "Call\n"
	             "./kmeans\n" << std::endl;
}


const cv::String keys =
    "{help h usage ? |                    | Hepl info                   }"
    "{@filename      | kmeans_test.png    | Input image.                }"
    "{output_path    | ./                 | Output_path.                }"
    "{nclusters      | 12                 | Default number of clusters. }"
    ;


cv::Mat showed_img;
std::vector<cv::Point> centers;
std::vector<double> centers_color;
size_t work_counter = 0;

//////////////////////////////////////////////////////////////////////////////
static void onMouse(int event, int x, int y, int, void*)
{
	if (x < 0 || x >= showed_img.cols || y < 0 ||
	    y >= showed_img.rows || event != cv::EVENT_LBUTTONDOWN)
		return;

	if (centers.size() == MAX_CLUSTERS) {
		const std::string message = "n_clusters <= " + std::to_string(MAX_CLUSTERS);
		const auto messTextSize = cv::getTextSize(message, CV_FONT_ITALIC, 0.8, 1, nullptr);
		cv::putText(showed_img, message, {10, 10 + messTextSize.height}, CV_FONT_ITALIC, 0.8, 255);
	} else {
		cv::Point point(x, y);
		cv::circle(showed_img, point, 5, 255, CV_FILLED);
		centers.push_back(point);
	}

	imshow(CLS_IMG, showed_img);
}

//////////////////////////////////////////////////////////////////////////////
void init_centers(const cv::Mat &image, const size_t &def_nclusters)
{
	if (centers.empty()) {
		std::cout << "Random centers are set." << std::endl;
		centers.resize(def_nclusters);
		std::generate(centers.begin(), centers.end(),
		              [&image](){return cv::Point(std::rand() % image.cols,
		                                          std::rand() % image.rows); });
	}

	for (const auto &point : centers)
		centers_color.push_back(image.at<uint8_t>(point.y, point.x, 0));
}

//////////////////////////////////////////////////////////////////////////////
bool update_centers(std::vector<cv::Point> &new_centers,
                    std::vector<double> &new_color,
                    const std::vector<size_t> &counter)
{
	auto div_fun = [](const cv::Point &point,
	                  const double &den) { return point / den; };
	std::transform(new_centers.begin(), new_centers.end(),
	               counter.begin(), new_centers.begin(), div_fun);
	std::transform(new_color.begin(), new_color.end(),
	               counter.begin(), new_color.begin(), std::divides<double>());

	if (cv::norm(new_centers, centers) > 1)
		return false;

	return true;
}

//////////////////////////////////////////////////////////////////////////////
double get_distance(const cv::Mat &image, const size_t &i,
                    const size_t &j, uint8_t &curr_cluster)
{
	double curr_distance = std::numeric_limits<double>::max();
	curr_cluster = 0;

	for (size_t ci = 0; ci < centers.size(); ++ci) {
		double temp_distance = 0;

		double pixel_color = image.at<uint8_t>(i, j, 0);
		double center_color = centers_color[ci];

		// Weight of distance
		temp_distance += SQR(((double)centers[ci].y - i) / image.rows);
		temp_distance += SQR(((double)centers[ci].x - j) / image.cols);

		// Weight of color
		temp_distance += 2 * SQR((pixel_color - center_color) / 255.);
		temp_distance = std::sqrt(temp_distance);

		if (temp_distance < curr_distance) {
			curr_distance = temp_distance;
			curr_cluster = ci;
		}
	}

	return curr_distance;
}

//////////////////////////////////////////////////////////////////////////////
void save_result(const cv::Mat &output_image,
                 const std::string &output_path,
                 const size_t &n_iter)
{
	std::string file_name =
	        output_path + "kmeans_" + std::to_string(work_counter) +
	        "_" + std::to_string(n_iter) + ".png";
	try {
		cv::imwrite(file_name, output_image);
	} catch (...) {
		std::cout << "Could not write image: " << file_name << std::endl;
	}
}

//////////////////////////////////////////////////////////////////////////////
void clustering(const cv::Mat &input_image,
                const std::string &output_path,
                const size_t &def_nclusters)
{
	work_counter++;

	std::chrono::time_point<std::chrono::system_clock> start, end;
	double elapsed_time{};

	cv::Mat gray_image;
	cvtColor(input_image, gray_image, CV_BGR2GRAY);

	init_centers(gray_image, def_nclusters);

	cv::Mat mask = cv::Mat::zeros(gray_image.rows, gray_image.cols, CV_8UC1);
	cv::Mat dst = cv::Mat::zeros(gray_image.rows, gray_image.cols, CV_64FC1);

	for (size_t itr = 1; ; ++itr) {
		// Start
		start = std::chrono::system_clock::now();

		dst = std::numeric_limits<double>::max();

		std::vector<size_t> counter(centers.size());
		std::vector<cv::Point> new_centers(centers.size());
		std::vector<double> new_color(centers.size());

		uint8_t cluster_num{};
		double distance{};

		// Assignment step
		for (int i = 0; i < gray_image.rows; ++i) {
			for (int j = 0; j < gray_image.cols; ++j) {
				distance = get_distance(gray_image, i, j, cluster_num);

				dst.at<double>(i, j) = distance;
				mask.at<uint8_t>(i, j) = cluster_num;

				counter[cluster_num]++;
				new_centers[cluster_num] += cv::Point(j, i);
				new_color[cluster_num] += gray_image.at<uint8_t>(i, j, 0);
			}
		}

		// Update new centers
		bool exit_flag = update_centers(new_centers, new_color, counter);

		// Set new centers
		centers = std::move(new_centers);
		centers_color = std::move(new_color);

		// End
		end = std::chrono::system_clock::now();
		elapsed_time += std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

		std::cout << "Itr #" << itr << ". All elapsed time: " <<
		             elapsed_time / 1000. << "s" << std::endl;

		// Show result
		cv::Mat buffer;
		cv::cvtColor(mask, buffer, cv::COLOR_GRAY2BGR);

		buffer *= 255.0 / centers.size();

		for (const auto &point : centers)
			cv::circle(buffer, point, 5, {0, 255, 0}, CV_FILLED);

		const std::string message = MSK_IMF(itr);
		cv::putText(buffer, message, {10, 10}, CV_FONT_ITALIC, 0.4, {0, 0, 255});

		cv::imshow("Result", buffer);

		// Write image
		save_result(buffer, output_path, itr);

		// Parse keys
		auto key = cvWaitKey();

		if (key == 'q' || key == 'Q' || key == 27) {
			std::cout << "Exit from current demonstration." << std::endl;
			break;
		}

		if (key == 'n' || key == 'N') {
			std::cout << "Next step of current demonstration." << std::endl;
			continue;
		}

		if (itr >= MAX_ITR) {
			std::cout << "Segmentation ended because the maximum number of iterations: "
			          << MAX_ITR << std::endl;
			break;
		}

		if (exit_flag) {
			std::cout << "Segmentation ended." << std::endl;
			break;
		}
	}

	cv::destroyWindow("Result");
}

//////////////////////////////////////////////////////////////////////////////
int kmeans(int argc, char** argv)
{
	cv::CommandLineParser parser(argc, argv, keys);

	if (parser.has("help")) {
		help();
		parser.printMessage();
		return 0;
	}

	const std::string filename = parser.get<std::string>(0);
	const std::string output_path = parser.get<std::string>("output_path");
	const size_t def_nclusters = parser.get<size_t>("nclusters");

	cv::Mat input_image = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	if (input_image.empty()) {
		std::cerr << "Could not open input image: " << filename
		          << std::endl;
		return -1;
	}

	while (true) {
		input_image.copyTo(showed_img);
		imshow(CLS_IMG, showed_img);
		cv::setMouseCallback(CLS_IMG, onMouse, 0);

		auto key = cv::waitKey();

		// Parse keys
		switch (key) {
			case 27:
			case 'q':
			case 'Q':
				std::cout << "Exit from program." << std::endl;
				return 0;
			case ' ':
			case 'l':
			case 'L':
				std::cout << "Demonstration launched." << std::endl;
				clustering(input_image, output_path, def_nclusters);
				centers.clear();
				centers_color.clear();
				continue;
			case 'r':
			case 'R':
			default:
				std::cout << "Demonstration restored." << std::endl;
				centers.clear();
				continue;
		}
	}

	return 0;
}
