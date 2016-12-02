#include "stdafx.h"

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



#define M_PI 3.14159265358979323846

size_t num_octaves;
size_t num_images;
double init_sigma;
double k;
double r_th = 10;
double dog_th = 0.03;


struct SIFTPoint {

	SIFTPoint(double i, double j, double sigm) :
		row{ i }, col{ j }, sigma{ sigm } {}

	double row;
	double col;
	double sigma;

	std::vector<double> rotation;
	cv::Point position; // actua pisition in image
	std::vector<double> descriptor;
};


static void help()
{
	std::cout << "\n This program demonstrates work of SIFT algorithm."
		<< std::endl;
}

const cv::String keys =
"{help h usage ? |               | Hepl info                         }"
"{@ref_image     | img/ref.jpg   | Reference image.                  }"
"{@test_image    | img/test1.jpg | Test image.                       }"
"{num_octaves    | 4             | Number of octaves in the pyramid. }"
"{num_images     | 5             | Number images in octave           }"
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
	std::vector<SIFTPoint> &points,
	double sigma)
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
				points.push_back(SIFTPoint(i, j, sigma));

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
	std::vector<std::vector<cv::Mat>> &dof_piramid,
	std::vector<std::vector<cv::Mat>> &blurred_piramid)
{
	cv::Size image_size = image.size();

	//std::vector<std::vector<cv::Mat>> blurred_piramid(num_octaves);
	blurred_piramid.resize(num_octaves);

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


void get_points(std::vector<std::vector<SIFTPoint>> &points,
	const std::vector<std::vector<cv::Mat>> &dof_piramid)
{
	points.resize(num_octaves);

	for (size_t i = 0; i < dof_piramid.size(); ++i) {
		//for (size_t i = 1; i < 3; ++i) {
		for (size_t j = 1; j < dof_piramid[i].size() - 1; ++j) {
			find_points(dof_piramid[i][j - 1], dof_piramid[i][j], dof_piramid[i][j + 1], points[i], j);
		}
	}
}


cv::Mat get_dD(const cv::Mat &bottom_mat,
	const cv::Mat &central_mat,
	const cv::Mat &top_mat,
	const SIFTPoint &point)
{
	cv::Mat res = cv::Mat(3, 1, CV_64F);

	res.at<double>(0, 0) = ((double)central_mat.at<uint8_t>(point.row + 1, point.col) -
		(double)central_mat.at<uint8_t>(point.row - 1, point.col)) / 2.0f;
	res.at<double>(1, 0) = ((double)central_mat.at<uint8_t>(point.row, point.col + 1) -
		(double)central_mat.at<uint8_t>(point.row, point.col - 1)) / 2.0f;
	res.at<double>(2, 0) = ((double)bottom_mat.at<uint8_t>(point.row, point.col) -
		(double)top_mat.at<uint8_t>(point.row, point.col)) / 2.0f;

	return std::move(res);
}


cv::Mat get_d2D(const cv::Mat &bottom_mat,
	const cv::Mat &central_mat,
	const cv::Mat &top_mat,
	const SIFTPoint &point)
{
	cv::Mat res = cv::Mat(3, 3, CV_64F);

	res.at<double>(0, 0) = ((double)central_mat.at<uint8_t>(point.row - 1, point.col) -
		2.0 * central_mat.at<uint8_t>(point.row, point.col) +
		(double)central_mat.at<uint8_t>(point.row + 1, point.col)) / 4.0;
	res.at<double>(1, 1) = ((double)central_mat.at<uint8_t>(point.row, point.col - 1) -
		2.0 * central_mat.at<uint8_t>(point.row, point.col) +
		(double)central_mat.at<uint8_t>(point.row, point.col + 1)) / 4.0;
	res.at<double>(2, 2) = ((double)bottom_mat.at<uint8_t>(point.row, point.col) -
		2.0 * central_mat.at<uint8_t>(point.row, point.col) +
		(double)top_mat.at<uint8_t>(point.row, point.col)) / 4.0;

	res.at<double>(0, 1) = ((double)central_mat.at<uint8_t>(point.row + 1, point.col + 1) -
		(double)central_mat.at<uint8_t>(point.row + 1, point.col - 1) -
		(double)central_mat.at<uint8_t>(point.row - 1, point.col + 1) +
		(double)central_mat.at<uint8_t>(point.row - 1, point.col - 1)) / 4.0;
	res.at<double>(1, 0) = res.at<double>(0, 1);

	res.at<double>(0, 2) = ((double)top_mat.at<uint8_t>(point.row + 1, point.col) +
		(double)top_mat.at<uint8_t>(point.row - 1, point.col) -
		(double)bottom_mat.at<uint8_t>(point.row + 1, point.col) +
		(double)bottom_mat.at<uint8_t>(point.row - 1, point.col)) / 4.0;
	res.at<double>(2, 0) = res.at<double>(0, 2);

	res.at<double>(1, 2) = ((double)-top_mat.at<uint8_t>(point.row, point.col + 1) +
		(double)top_mat.at<uint8_t>(point.row, point.col - 1) -
		(double)bottom_mat.at<uint8_t>(point.row, point.col + 1) +
		(double)bottom_mat.at<uint8_t>(point.row, point.col - 1)) / 4.0;
	res.at<double>(2, 1) = res.at<double>(1, 2);

	return std::move(res);
}


cv::Mat get_H_from_d2D(const cv::Mat& d2D)
{
	cv::Mat res(2, 2, CV_64F);
	res.at<double>(0, 0) = d2D.at<double>(0, 0);
	res.at<double>(0, 1) = d2D.at<double>(0, 1);
	res.at<double>(1, 0) = d2D.at<double>(1, 0);
	res.at<double>(1, 1) = d2D.at<double>(1, 1);

	return std::move(res);
}


bool check_point(const cv::Mat &bottom_mat,
	const cv::Mat &central_mat,
	const cv::Mat &top_mat,
	SIFTPoint &point)
{
	cv::Mat dD, d2D, dX;

	while (true) {
		dD = get_dD(bottom_mat, central_mat, top_mat, point);
		d2D = get_d2D(bottom_mat, central_mat, top_mat, point);

		dX = -d2D.inv() * dD;

		point.row += std::round(dX.at<double>(0, 0));
		point.col += std::round(dX.at<double>(1, 0));
		point.sigma += std::round(dX.at<double>(2, 0));

		if (point.sigma >= num_images ||
			point.sigma <= 0 ||
			point.row >= central_mat.rows ||
			point.row <= 0 ||
			point.col >= central_mat.cols ||
			point.col <= 0)
			return false;

		if (dX.at<double>(0, 0) < 0.5 ||
			dX.at<double>(1, 0) < 0.5 ||
			dX.at<double>(2, 0) < 0.5)
			break;
	}

	//cv::transpose(dD, dD);
	//double value_D = central_mat.at<double>(point.row, point.col) + 0.5 * dD.t() * dX;
	double value_D = central_mat.at<uint8_t>(point.row, point.col) + 0.5 * dD.dot(dX);

	if (std::abs(value_D) < dog_th)
		return false;

	cv::Mat H = get_H_from_d2D(d2D);

	if ((cv::trace(H) / cv::determinant(H))[0] >= ((r_th + 1)*(r_th + 1) / (r_th)))
		return false;

	return true;
}


void filt_point(std::vector<std::vector<SIFTPoint>> &points,
	const std::vector<std::vector<cv::Mat>> &dof_piramid)
{
	for (size_t i = 0; i < points.size(); ++i) {

		size_t j = 0;

		while (j < points[i].size()) {

			bool delete_point = !check_point(dof_piramid[i][points[i][j].sigma - 1],
				dof_piramid[i][points[i][j].sigma],
				dof_piramid[i][points[i][j].sigma + 1],
				points[i][j]);

			if (!delete_point)
				j++;
			else
				points[i].erase(points[i].begin() + j);
		}
	}
}


void update_rotation_point(std::vector<double>& hist, SIFTPoint& point, long& i)
{
	double prev, curr, next;
	curr = hist[i];
	if (i - 1 >= 0)
		prev = hist[i - 1];
	else
		prev = hist[hist.size() - 1];
	if (i + 1 < hist.size())
		next = hist[i + 1];
	else
		next = hist[0];

	point.rotation.push_back(i*10.0 +
		((next - prev) / 20.0) /
		((next - 2.0 * curr + prev + 1e-6) / 400.0));
}


void cacl_hist(const cv::Mat &log, SIFTPoint &point, std::vector<double> &hist)
{
	double R = 3 * init_sigma * point.sigma;

	long init_i, end_i, init_j, end_j;

	init_i = point.row - R;
	if (init_i < 1)
		init_i = 1;
	init_j = point.col - R;
	if (init_j < 1)
		init_j = 1;

	end_i = point.row + R;
	if (init_i >= log.rows - 1)
		init_i = log.rows - 2;
	end_j = point.col + R;
	if (init_j >= log.cols - 1)
		init_j = log.cols - 2;

	for (size_t i = init_i; i < end_i; ++i) {
		for (size_t j = init_j; j < end_j; ++j) {
			double curr_R = std::sqrt(std::pow(i - point.row, 2.0)
				+ std::pow(j - point.col, 2.0));
			if (curr_R <= R) {
				double theta = std::atan((log.at<uint8_t>(i, j + 1)
					- log.at<uint8_t>(i, j - 1))
					/ (log.at<uint8_t>(i + 1, j)
						- log.at<uint8_t>(i - 1, j) + 1e-6));
				double m = std::sqrt(std::pow(log.at<uint8_t>(i, j + 1)
					- log.at<uint8_t>(i, j - 1), 2.0)
					+ std::pow(log.at<uint8_t>(i + 1, j)
						- log.at<uint8_t>(i - 1, j), 2.0));

				hist[(int)((theta + 3.14 / 2.0) * 180.0 / 3.14 / 10.0)] += m;
			}
		}
	}
}


void calc_rotation(const cv::Mat &log, SIFTPoint &point)
{
	std::vector<double> hist(36, 0);
	cacl_hist(log, point, hist);

	long max_i = std::distance(hist.begin(), std::max_element(hist.begin(), hist.end()));
	update_rotation_point(hist, point, max_i);

	for (long i = 0; i < hist.size(); ++i)
		if (hist[i] > 0.8 * hist[max_i])
			if ((i != max_i)
				&& (i - 1 != max_i)
				&& (i + 1 != max_i))
				update_rotation_point(hist, point, i);
}


void get_rotation_point(std::vector<std::vector<SIFTPoint>> &points,
	const std::vector<std::vector<cv::Mat>> &log_piramid)
{
	for (size_t i = 0; i < points.size(); ++i)
		for (size_t j = 0; j < points.size(); ++j)
			calc_rotation(log_piramid[i][std::round(points[i][j].sigma)], points[i][j]);
}


void compute_descriptors(std::vector<std::vector<SIFTPoint>>& points, const cv::Mat& image)
{
	int k = 0, n = 0;
	for (size_t i = 0; i < points.size(); ++i) {
		for (size_t j = 0; j < points[i].size(); ++j) {

			cv::Point position = cv::Point(points[i][j].col, points[i][j].row) * int(i + 1); // actual position in image

			if (position.x > 7 && position.y > 7 && position.x < image.cols - 7 && position.y < image.rows - 7) {

				n++;
				points[i][j].position = position;
				points[i][j].descriptor.resize(128, 0);

				// 16x16 neighborhood
				cv::Mat roi = image(cv::Rect(position - cv::Point(8, 8), cv::Size(16, 16)));

				// 4x4 subregions
				for (size_t bx = 0; bx < 4; bx++) {
					for (size_t by = 0; by < 4; by++) {

						// 8-bin histogram
						for (size_t sx = 0; sx < 4; sx++) {
							for (size_t sy = 0; sy < 4; sy++) {

								size_t x = 4 * bx + sx; // coordinates in roi
								size_t y = 4 * by + sy;

								double dx, dy; // derivatives
								if (x == 0)
									dx = double(roi.at<uint8_t>(x + 1, y)) - double(roi.at<uint8_t>(x, y)); // left border
								else if (x == 15)
									dx = double(roi.at<uint8_t>(x, y)) - double(roi.at<uint8_t>(x - 1, y)); // right border
								else
									dx = 0.5*double(roi.at<uint8_t>(x + 1, y)) - 0.5*double(roi.at<uint8_t>(x - 1, y));

								if (y == 0)
									dy = double(roi.at<uint8_t>(x, y + 1)) - double(roi.at<uint8_t>(x, y)); // top border
								else if (y == 15)
									dy = double(roi.at<uint8_t>(x, y)) - double(roi.at<uint8_t>(x, y - 1)); // bottom border
								else
									dy = 0.5*double(roi.at<uint8_t>(x, y + 1)) - 0.5*double(roi.at<uint8_t>(x, y - 1));

								double orientation;
								try {
									// (-pi,pi) -> (0,2*pi)
									orientation = atan2(dy, dx);
									if (orientation < 0)
										orientation = 2 * M_PI + orientation;
								}
								catch (...) { // dx too small
									orientation = 0;
								}
								double magnitude = sqrt(dx*dx + dy*dy);

								// weight with gaussian
								double s = points[i][j].sigma;
								double rx = double(x) - 8;
								double ry = double(y) - 8;
								magnitude *= exp(-rx*rx / (s*s))*exp(-ry*ry / (s*s));

								uint8_t index = uint8_t(4 * orientation / M_PI);
								points[i][j].descriptor[(by * 4 + bx) * 8 + index] += magnitude; // increment histogram
							}
						}
					}
				}

				// normalization, cut values > 0.2 until they exists
				bool flag = true;
				int m = 0;
				while (flag) {
					if (m == 300)
						break;
					flag = false;
					double s = 0;
					for (size_t d = 0; d < points[i][j].descriptor.size(); d++)
						s += points[i][j].descriptor[d] * points[i][j].descriptor[d];
					s = sqrt(s); // norm
					if (s > 0.95)
						break;
					for (size_t d = 0; d < points[i][j].descriptor.size(); d++) {
						points[i][j].descriptor[d] /= s;
						if (points[i][j].descriptor[d] > 0.2) {
							points[i][j].descriptor[d] = 0.19999;
							flag = true;
						}
					}
					m++;
					std::cout << "i=" << i << ", j=" << j << ", m=" << m << std::endl;
				}
			}
			else
			{
				// roi out of matrix, erase point
				points[i].erase(points[i].begin() + j);
				j--;
				k++;
			}
		}
	}
	std::cout << "out of matrix " << k << " in: " << n << std::endl;
}


// double euclidian_distance(const std::vector<double> v1, const std::vector<double> v2)
double euclidian_distance(const SIFTPoint& p1, const SIFTPoint& p2)
{
	// TODO: check watever vector sizes is equal
	double d = 0;
	for (size_t i = 0; i < p1.descriptor.size(); i++)
		d += (p1.descriptor[i] - p2.descriptor[i])*(p1.descriptor[i] - p2.descriptor[i]);
	return sqrt(d);
}


std::vector<std::tuple<size_t, size_t, double>> compute_pairs(const std::vector<SIFTPoint>& points1, const std::vector<SIFTPoint>& points2)
{

	

	std::vector<std::tuple<size_t, size_t, double>> pairs;
	for (size_t i = 0; i < points1.size(); i++) {

		double min_distance = 10;
		size_t min_distance_index = 0;
		for (size_t j = 0; j < points2.size(); j++) {

			double d = euclidian_distance(points1[i], points2[j]);
			if (d < min_distance) {
				min_distance = d;
				min_distance_index = j;
			}
		}

		pairs.push_back(std::make_tuple(i, min_distance_index, min_distance));
	}


	return std::move(pairs);
}
// {
	/*
	double min_distance = 10;
	size_t min_distance_i = 0;
	size_t min_distance_j = 0;

	auto point = ref_points[0][0];
	for (size_t it = 0; it < test_points.size(); ++it) {
		for (size_t jt = 0; jt < test_points[it].size(); ++jt) {
			// calculate distance
			size_t l = test_points[it][jt].descriptor_v.size();
			if (l == 0)
				continue;
			double d = 0;
			for (size_t b = 0; b < test_points[it][jt].descriptor_v.size(); b++) {
				double t = point.descriptor_v[b] - test_points[it][jt].descriptor_v[b];
				d += t*t;
			}
			d = sqrt(d);
			if (d < min_distance) {
				min_distance = d;
				min_distance_i = it;
				min_distance_j = jt;
			}
		}
	}
	cv::circle(ref_image, cv::Point(ref_points[0][0].col, ref_points[0][0].row), 5, cv::Scalar(0, 0, 255), 5);
	cv::circle(test_image, cv::Point(test_points[min_distance_i][min_distance_j].col, test_points[min_distance_i][min_distance_j].row), 5, cv::Scalar(0, 0, 255), 5);
	*/
//}

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

	int start;
	double elapsed_time;

	// Load images
	cv::Mat ref_image;
	cv::Mat test_image;

	if (!load_image(ref_image, ref_file))
		return;
	if (!load_image(test_image, test_file))
		return;

	// RGB2GRAY
	cv::Mat g_ref_image;
	cv::Mat g_test_image;

	cv::cvtColor(ref_image, g_ref_image, cv::COLOR_RGB2GRAY);
	cv::cvtColor(test_image, g_test_image, cv::COLOR_RGB2GRAY);

	std::cout << "End RGB to Gray conversations." << std::endl;


	// DoG piramids
	std::vector<std::vector<cv::Mat>> ref_dog_piramid;
	std::vector<std::vector<cv::Mat>> test_dog_piramid;
	std::vector<std::vector<cv::Mat>> ref_log_piramid;
	std::vector<std::vector<cv::Mat>> test_log_piramid;

	get_dog_piramid(g_ref_image, ref_dog_piramid, ref_log_piramid);
	get_dog_piramid(g_test_image, test_dog_piramid, test_log_piramid);

	std::cout << "End get DoG pyramids." << std::endl;

	// Find points
	start = clock();
	std::vector<std::vector<SIFTPoint>> ref_points;
	std::vector<std::vector<SIFTPoint>> test_points;

	get_points(ref_points, ref_dog_piramid);
	get_points(test_points, test_dog_piramid);
	elapsed_time = (clock() - start) / (1.0*CLOCKS_PER_SEC);

	std::cout << "End get points. " << elapsed_time << " sec" << std::endl;

	start = clock();
	filt_point(ref_points, ref_dog_piramid);
	filt_point(test_points, test_dog_piramid);
	elapsed_time = (clock() - start) / (1.0*CLOCKS_PER_SEC);

	std::cout << "End filtration of points. " << elapsed_time << " sec" << std::endl;

	start = clock();
	get_rotation_point(ref_points, ref_log_piramid);
	get_rotation_point(test_points, test_log_piramid);
	elapsed_time = (clock() - start) / (1.0*CLOCKS_PER_SEC);

	std::cout << "End get rotation of points. " << elapsed_time << " sec" << std::endl;


	// Build descriptors
	start = clock();
	compute_descriptors(ref_points, g_ref_image);
	compute_descriptors(test_points, g_test_image);
	elapsed_time = (clock() - start) / (1.0*CLOCKS_PER_SEC);

	std::cout << "End building descriptors. " << elapsed_time << " sec" << std::endl;


	// TODO: Compare images
	start = clock();
	std::vector<SIFTPoint> ref_vector, test_vector;

	for (size_t i = 0; i < ref_points.size(); i++)
		for (size_t j = 0; j < ref_points[i].size(); j++)
			ref_vector.push_back(ref_points[i][j]);

	for (size_t i = 0; i < test_points.size(); i++)
		for (size_t j = 0; j < test_points[i].size(); j++)
			test_vector.push_back(test_points[i][j]);

	auto pairs = compute_pairs(ref_vector, test_vector);
	elapsed_time = (clock() - start) / (1.0*CLOCKS_PER_SEC);

	std::cout << "End comparing. " << elapsed_time << " sec" << std::endl;



	// TODO: Show result

	cv::imshow("Reference image", ref_image);
	cv::imshow("Test image", test_image);
	cv::imshow("Test result", ref_dog_piramid[0][0]);

	cv::waitKey(0);

	cv::destroyAllWindows();
}
