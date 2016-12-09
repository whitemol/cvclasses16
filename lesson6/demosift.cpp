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
#include <algorithm>



#define M_PI 3.14159265358979323846

using IMG_TYPE = float;

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
	cv::Point position; // actual pisition in image
	std::vector<double> descriptor; // float? conversion to CV_32F will be required anyway
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


std::tuple<IMG_TYPE, IMG_TYPE> extr_win(const cv::Mat &mat1,
	const cv::Mat &mat2,
	const cv::Mat &mat3,
	const size_t &row, const size_t &col)
{
	IMG_TYPE max_val = std::numeric_limits<IMG_TYPE>::min();
	IMG_TYPE min_val = std::numeric_limits<IMG_TYPE>::max();

	for (size_t i = row - 1; i <= row + 1; ++i)
		for (size_t j = col - 1; j <= col + 1; ++j) {
			if (i != row && j != col) {
				max_val = std::max(max_val, mat1.at<IMG_TYPE>(i, j));
				max_val = std::max(max_val, mat2.at<IMG_TYPE>(i, j));
				max_val = std::max(max_val, mat3.at<IMG_TYPE>(i, j));

				min_val = std::min(min_val, mat1.at<IMG_TYPE>(i, j));
				min_val = std::min(min_val, mat2.at<IMG_TYPE>(i, j));
				min_val = std::min(min_val, mat3.at<IMG_TYPE>(i, j));
			}
		}

	max_val = std::max(max_val, mat1.at<IMG_TYPE>(row, col));
	max_val = std::max(max_val, mat3.at<IMG_TYPE>(row, col));

	min_val = std::min(min_val, mat1.at<IMG_TYPE>(row, col));
	min_val = std::min(min_val, mat3.at<IMG_TYPE>(row, col));

	return std::tuple<IMG_TYPE, IMG_TYPE>(max_val, min_val);
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
			IMG_TYPE value = central_mat.at<IMG_TYPE>(i, j);

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


void get_dog_piramid(cv::Mat& image,
	std::vector<std::vector<cv::Mat>> &dof_piramid,
	std::vector<std::vector<cv::Mat>> &blurred_piramid)
{
	cv::Size image_size = image.size();
	//cv::Mat img; image.convertTo(img, CV_32F); image = img;
	image.convertTo(image, CV_32F);

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

	res.at<double>(0, 0) = ((double) central_mat.at<IMG_TYPE>(point.row + 1, point.col) -
		(double)central_mat.at<IMG_TYPE>(point.row - 1, point.col)) / 2.0f;
	res.at<double>(1, 0) = ((double) central_mat.at<IMG_TYPE>(point.row, point.col + 1) -
		(double)central_mat.at<IMG_TYPE>(point.row, point.col - 1)) / 2.0f;
	res.at<double>(2, 0) = ((double) top_mat.at<IMG_TYPE>(point.row, point.col) -
		(double) bottom_mat.at<IMG_TYPE>(point.row, point.col)) / 2.0f;

	return std::move(res);
}


cv::Mat get_d2D(const cv::Mat &bottom_mat,
	const cv::Mat &central_mat,
	const cv::Mat &top_mat,
	const SIFTPoint &point)
{
	cv::Mat res = cv::Mat(3, 3, CV_64F);

	res.at<double>(0, 0) = ((double)central_mat.at<IMG_TYPE>(point.row - 1, point.col) -
		2.0 * central_mat.at<IMG_TYPE>(point.row, point.col) +
		(double)central_mat.at<IMG_TYPE>(point.row + 1, point.col));
	res.at<double>(1, 1) = ((double)central_mat.at<IMG_TYPE>(point.row, point.col - 1) -
		2.0 * central_mat.at<IMG_TYPE>(point.row, point.col) +
		(double)central_mat.at<IMG_TYPE>(point.row, point.col + 1));
	res.at<double>(2, 2) = ((double)bottom_mat.at<IMG_TYPE>(point.row, point.col) -
		2.0 * central_mat.at<IMG_TYPE>(point.row, point.col) +
		(double)top_mat.at<IMG_TYPE>(point.row, point.col));

	res.at<double>(0, 1) = ((double)central_mat.at<IMG_TYPE>(point.row + 1, point.col + 1) -
		(double)central_mat.at<IMG_TYPE>(point.row + 1, point.col - 1) -
		(double)central_mat.at<IMG_TYPE>(point.row - 1, point.col + 1) +
		(double)central_mat.at<IMG_TYPE>(point.row - 1, point.col - 1)) / 4.0;
	res.at<double>(1, 0) = res.at<double>(0, 1);

	res.at<double>(0, 2) = ((double)top_mat.at<IMG_TYPE>(point.row + 1, point.col) +
		(double)top_mat.at<IMG_TYPE>(point.row - 1, point.col) -
		(double)bottom_mat.at<IMG_TYPE>(point.row + 1, point.col) +
		(double)bottom_mat.at<IMG_TYPE>(point.row - 1, point.col)) / 4.0;
	res.at<double>(2, 0) = res.at<double>(0, 2);

	res.at<double>(1, 2) = ((double)top_mat.at<IMG_TYPE>(point.row, point.col + 1) +
		(double)top_mat.at<IMG_TYPE>(point.row, point.col - 1) -
		(double)bottom_mat.at<IMG_TYPE>(point.row, point.col + 1) +
		(double)bottom_mat.at<IMG_TYPE>(point.row, point.col - 1)) / 4.0;
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
			point.sigma < 0 ||
			point.row >= central_mat.rows ||
			point.row < 0 ||
			point.col >= central_mat.cols ||
			point.col < 0 ||
			point.sigma < 0 ||
			point.sigma >= num_images)
			return false;

		if (dX.at<double>(0, 0) < 0.5 ||
			dX.at<double>(1, 0) < 0.5 ||
			dX.at<double>(2, 0) < 0.5)
			break;
		//else
		//	return false;
	}

	//cv::transpose(dD, dD);
	//double value_D = central_mat.at<double>(point.row, point.col) + 0.5 * dD.t() * dX;
	double value_D = central_mat.at<IMG_TYPE>(point.row, point.col) + 0.5 * dD.dot(dX);

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

	point.rotation.push_back(i*10.0 -
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
	if (end_i >= log.rows - 1)
		end_i = log.rows - 2;
	end_j = point.col + R;
	if (end_j >= log.cols - 1)
		end_j = log.cols - 2;

	for (size_t i = init_i; i < end_i; ++i) {
		for (size_t j = init_j; j < end_j; ++j) {
			double curr_R = std::sqrt(std::pow(i - point.row, 2.0)
				+ std::pow(j - point.col, 2.0));
			if (curr_R <= R) {
				double theta = std::atan((log.at<IMG_TYPE>(i, j + 1)
					- log.at<IMG_TYPE>(i, j - 1))
					/ (log.at<IMG_TYPE>(i + 1, j)
						- log.at<IMG_TYPE>(i - 1, j) + 1e-6));
				double m = std::sqrt(std::pow(log.at<IMG_TYPE>(i, j + 1)
					- log.at<IMG_TYPE>(i, j - 1), 2.0)
					+ std::pow(log.at<IMG_TYPE>(i + 1, j)
						- log.at<IMG_TYPE>(i - 1, j), 2.0));

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
		for (size_t j = 0; j < points[i].size(); ++j)
			calc_rotation(log_piramid[i][std::round(points[i][j].sigma)], points[i][j]);
}


void compute_descriptors(std::vector<std::vector<SIFTPoint>>& points, const cv::Mat& image)
{
	int k = 0, n = 0;
	for (size_t i = 0; i < points.size(); ++i) {
		for (size_t j = 0; j < points[i].size(); ++j) {

			cv::Point position = cv::Point(points[i][j].col * std::pow(2, i), points[i][j].row * std::pow(2, i));//int(i + 1); // actual position in image

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
									dx = double(roi.at<IMG_TYPE>(x + 1, y)) - double(roi.at<IMG_TYPE>(x, y)); // left border
								else if (x == 15)
									dx = double(roi.at<IMG_TYPE>(x, y)) - double(roi.at<IMG_TYPE>(x - 1, y)); // right border
								else
									dx = 0.5*double(roi.at<IMG_TYPE>(x + 1, y)) - 0.5*double(roi.at<IMG_TYPE>(x - 1, y));

								if (y == 0)
									dy = double(roi.at<IMG_TYPE>(x, y + 1)) - double(roi.at<IMG_TYPE>(x, y)); // top border
								else if (y == 15)
									dy = double(roi.at<IMG_TYPE>(x, y)) - double(roi.at<IMG_TYPE>(x, y - 1)); // bottom border
								else
									dy = 0.5*double(roi.at<IMG_TYPE>(x, y + 1)) - 0.5*double(roi.at<IMG_TYPE>(x, y - 1));

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
								orientation -= points[i][j].rotation[0] * M_PI / 180.0;
								if (orientation < 0)
									orientation = 2 * M_PI + orientation;

								double magnitude = sqrt(dx*dx + dy*dy);

								// weight with gaussian
								double s = points[i][j].sigma;
								double rx = double(x) - 8;
								double ry = double(y) - 8;
								magnitude *= exp(-rx*rx / (s*s))*exp(-ry*ry / (s*s));

								size_t index = size_t(4 * orientation / M_PI);
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
					// std::cout << "i=" << i << ", j=" << j << ", m=" << m << std::endl;
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
	// std::cout << "out of matrix " << k << " in: " << n << std::endl;
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


// very slow searching O(ref_size*test_size)
// tuple: index in test, index in ref, distance
std::vector<std::tuple<size_t, size_t, double>> compute_pairs(const std::vector<SIFTPoint>& ref, const std::vector<SIFTPoint>& test)
{
	std::vector<std::tuple<size_t, size_t, double>> pairs;

/*	std::vector<bool> get_test(test.size(), false);

	for (size_t i = 0; i < test.size(); i++) {
		double min_distance = 3; // all vectors are normalized -> norm of difference <= 2
		size_t min_distance_index = 0;
		for (size_t j = 0; j < ref.size(); j++) {
			if (get_test[j] == false) {
				double d = euclidian_distance(test[i], ref[j]);
				if (d < min_distance) {
					min_distance = d;
					min_distance_index = j;
				}
			}
		}

		if (min_distance <= 2) {
			pairs.push_back(std::make_tuple(i, min_distance_index, min_distance));
			get_test[min_distance_index] = true;
		}
	}*/

	for (size_t i = 0; i < test.size(); i++)
	{
		double min_distance = 3; // all vectors are normalized -> norm of difference <= 2
		size_t min_distance_index = 0;
		for (size_t j = 0; j < ref.size(); j++)
		{
			double d = euclidian_distance(test[i], ref[j]);
			if (d < min_distance) {
				min_distance = d;
				min_distance_index = j;
			}
		}

		pairs.push_back(std::make_tuple(i, min_distance_index, min_distance));
	}

	std::vector<std::tuple<size_t, size_t, double>> pairs_filt;
	std::vector<bool> use_vec(pairs.size(), false);
	bool del_ref, del_test;

	for (size_t i = 0; i < pairs.size(); i++) {

		del_ref = false;
		del_test = false;

		if (use_vec[i] == false) {
			pairs_filt.push_back(pairs[i]);

			SIFTPoint ref_first_i = ref[std::get<1>(pairs[i])]; // 0 test 1 ref
			SIFTPoint test_first_i = test[std::get<0>(pairs[i])]; // 0 test 1 ref

			for (size_t j = i; j < pairs.size(); j++) {

				SIFTPoint ref_first_j = ref[std::get<1>(pairs[j])];
				SIFTPoint test_first_j = test[std::get<0>(pairs[j])];

				if (ref_first_i.position == ref_first_j.position &&
					std::round(ref_first_i.sigma) == std::round(ref_first_j.sigma))
					use_vec[j] = true;

				if (test_first_i.position == test_first_j.position &&
					std::round(test_first_i.sigma) == std::round(test_first_j.sigma))
					use_vec[j] = true;
			}
		}
		use_vec[i] = true;
	}
	//auto pairs_filt = pairs;

	std::sort(pairs_filt.begin(), pairs_filt.end(),
			  [](std::tuple<size_t, size_t, double> t1,
			  std::tuple<size_t, size_t, double> t2) {return std::get<2>(t1) < std::get<2>(t2); });

	return std::move(pairs_filt);
	//return std::move(pairs);
}


std::vector<std::tuple<size_t, size_t, double>>
compute_pairs(const std::vector<std::vector<SIFTPoint>>& ref,
              const std::vector<std::vector<SIFTPoint>>& test)
{
	std::vector<std::vector<std::tuple<size_t, size_t, double>>> pairs(4);

	for (size_t k = 0; k < ref.size(); k++) {
		for (size_t i = 0; i < test[k].size(); i++) {
			double min_distance = 3; // all vectors are normalized -> norm of difference <= 2
			size_t min_distance_index = 0;
			for (size_t j = 0; j < ref[k].size(); j++) {
				double d = euclidian_distance(test[k][i], ref[k][j]);
				if (d < min_distance) {
					min_distance = d;
					min_distance_index = j;
				}
			}

			pairs[k].push_back(std::make_tuple(i, min_distance_index, min_distance));
		}
	}

	std::vector<std::tuple<size_t, size_t, double>> pairs_v;
	for (size_t k = 0; k < pairs.size(); k++) {
		for (size_t i = 0; i < pairs[k].size(); i++) {
			pairs_v.push_back(pairs[k][i]);
		}
	}

	std::sort(pairs_v.begin(), pairs_v.end(),
			  [](std::tuple<size_t, size_t, double> t1,
			  std::tuple<size_t, size_t, double> t2) {return std::get<2>(t1) < std::get<2>(t2); });

	return std::move(pairs_v);
}


void drawResult(const cv::Mat& ref_image, const cv::Mat& test_image, cv::Mat& result,
				const std::vector<std::tuple<size_t, size_t, double>>& pairs,
				const std::vector<SIFTPoint>& ref_vector,
				const std::vector<SIFTPoint>& test_vector,
				const size_t num_points_begin,
				const size_t num_points_end)
{
	result = cv::Mat(cv::Size(ref_image.cols + test_image.cols, std::max(ref_image.rows, test_image.rows)), ref_image.type());
	ref_image.copyTo(result(cv::Rect(0, 0, ref_image.cols, ref_image.rows)));
	test_image.copyTo(result(cv::Rect(ref_image.cols, 0, test_image.cols, test_image.rows)));
	size_t num_points = num_points_end - num_points_begin;

	for (size_t i = num_points_begin; i < num_points_end; ++i) { // pairs.size()
		auto first = pairs[i];
		SIFTPoint test_first = test_vector[std::get<0>(first)];
		SIFTPoint ref_first = ref_vector[std::get<1>(first)];

		long r = 255 * 3 / (num_points - i) - 255 * 2;
		long g = 255 * 3 / (num_points - i) - 255 * 1;
		long b = 255 * 3 / (num_points - i);
		if (b > 255) b = 255;
		if (r > 255) r = 255;
		if (g > 255) g = 255;
		auto color = cv::Scalar_<uint8_t>(r, g, b);

		cv::circle(result, ref_first.position, 5, color, 5);
		cv::circle(result, test_first.position + cv::Point(ref_image.cols, 0), 5, color, 5);
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


	// Compare images
	start = clock();

	std::vector<SIFTPoint> ref_vector, test_vector; // store in two vectors

	for (size_t i = 0; i < ref_points.size(); i++)
		for (size_t j = 0; j < ref_points[i].size(); j++)
			ref_vector.push_back(ref_points[i][j]);

	for (size_t i = 0; i < test_points.size(); i++)
		for (size_t j = 0; j < test_points[i].size(); j++)
			test_vector.push_back(test_points[i][j]);

	auto pairs = compute_pairs(ref_vector, test_vector); // tuple: index in test, index in ref, distance
	//auto pairs = compute_pairs(ref_points, test_points); // tuple: index in test, index in ref, distance
	elapsed_time = (clock() - start) / (1.0*CLOCKS_PER_SEC);

	// auto pairs = compute_pairs(ref_points, test_points);
	std::cout << "End comparing. " << elapsed_time << " sec" << std::endl;
	std::cout << "End comparing. Count of pairs: " << pairs.size() << std::endl;


	// TODO: Show result

	cv::Mat result; 
	drawResult(ref_image, test_image, result, pairs, ref_vector, test_vector, 0, 20);
	cv::namedWindow("result 20 f", cv::WINDOW_NORMAL);
	cv::imshow("result 20 f", result);

	drawResult(ref_image, test_image, result, pairs, ref_vector, test_vector, pairs.size() - 20, pairs.size());
	cv::namedWindow("result 20 l", cv::WINDOW_NORMAL);
	cv::imshow("result 20 l", result);

	drawResult(ref_image, test_image, result, pairs, ref_vector, test_vector, 0, pairs.size());
	cv::namedWindow("result all", cv::WINDOW_NORMAL);
	cv::imshow("result all", result);

	cv::waitKey(0);

	cv::destroyAllWindows();
}
