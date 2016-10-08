///@File: otsuthreshold.cpp
///@Brief: Contains an example using the otsu threshold algorithm.
///@Author: Alexander Budylev
///@Date: 4 October 2016

#include "stdafx.h"

#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

const cv::String keys =
  "{help h usage ? |               | Help info.   }"
  "{@filename      | otsu/wolf.jpg | Input image. }"
  "{out_folder     | otsu/         | Out folder.  }"
;

static void help()
{
  cout << "\nThis program demonstrates the otsu segmentation algorithm\n"
    "Usage:\n"
    "./otsu [image_name (default is otsu/wolf.jpg) out_folder (default is otsu/)]\n" << endl;
  cout << "Hot keys: \n"
    "\tESC - quit the program\n"
    "\tr - restore the original image\n"
    "\tw or SPACE - run otsu segmentation algorithm\n"
    "\tn - continue otsu segmentation algorithm\n";
}

Mat img_hist, img_gray, th_img;
double threshold_value, varMax, sum_hist, wB, wF, sumB, elapsed_time;
bool flag_end;
int iter_counter;

void save_img(const std::string& out_folder) {
  std::string file_name =
    out_folder + "otsu_threshold" + std::to_string(iter_counter) + ".png";
  try {
    cv::imwrite(file_name, th_img);
  }
  catch (...) {
    std::cout << "Could not write image: " << file_name << std::endl;
  }
}

void save_hist(const std::string& out_folder) {
  std::string file_hist_name = out_folder + "otsu_hist.png";
  try {
    cv::imwrite(file_hist_name, img_hist);
  }
  catch (...) {
    std::cout << "Could not write image: " << file_hist_name << std::endl;
  }
}

void show_hist() {
  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound((double)hist_w / 256.0);
  Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

  /// Normalize the result to [ 0, histImage.rows ]
  Mat hist_norm;
  normalize(img_hist, hist_norm, 0, histImage.rows, NORM_MINMAX, -1, Mat());

  for (int i = 1; i < 256; i++)
    line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist_norm.at<float>(i - 1))),
         Point(bin_w*(i), hist_h - cvRound(hist_norm.at<float>(i))),
         Scalar(255, 0, 0), 2, 8, 0);

  circle(histImage, Point(bin_w*(threshold_value),
         hist_h - cvRound(hist_norm.at<float>(threshold_value))),
         2, Scalar(0, 255, 0), FILLED, LINE_AA);
  /// Display
  imshow("Histogram of image.", histImage);
}

void show_result() {
  imshow("Thresholded image.", th_img);
  show_hist();
}

void make_result() {
  threshold(img_gray, th_img, threshold_value, 255, CV_THRESH_BINARY);
}

void calc_threshold() {
  if (flag_end)
    return;

  chrono::time_point<chrono::system_clock> start, end;

  start = chrono::system_clock::now();

  for (int t = static_cast<int>(threshold_value + 1); t < 256; t++) {

    wB += img_hist.at<float>(t); // Weight Background
    if (wB == 0 || img_hist.at<float>(t) == 0)
      continue;

    wF = img_gray.total() - wB; // Weight Foreground
    if (wF == 0)
      break;

    sumB += t * img_hist.at<float>(t);

    double mB = sumB / wB; // Mean Background
    double mF = (sum_hist - sumB) / wF; // Mean Foreground

    // Calculate Between Class Variance
    double varBetween = wB * wF * (mB - mF) * (mB - mF);

    // Check if new maximum found
    if (varBetween > varMax) {
      varMax = varBetween;
      threshold_value = t;
      iter_counter++;
      break;
    }

    if (t == 255) {
      flag_end = true;
    }
  }

  // End
  end = chrono::system_clock::now();
  elapsed_time += chrono::duration_cast<chrono::milliseconds>(end - start).count();

  cout << "Iteration number" << iter_counter << ". All elapsed time is: " <<
    elapsed_time / 1000. << "seconds." << endl;

  make_result();
  show_result();
}

void init_otsu_values() {
  threshold_value = 0;
  varMax = 0;
  wB = 0;
  wF = 0;
  sumB = 0;
  flag_end = false;
  iter_counter = 0;
  elapsed_time = 0;
}

void init_values(Mat input_image) {
  cvtColor(input_image, img_gray, COLOR_RGB2GRAY);

  chrono::time_point<chrono::system_clock> start, end;

  start = chrono::system_clock::now();

  int histSize = 256;
  float range[] = { 0, 256 };
  const float* histRange = { range };
  calcHist(&img_gray, 1, 0, Mat(), img_hist, 1, &histSize, &histRange, true, false);

  sum_hist = 0;
  for (int t = 0; t < 256; t++)
    sum_hist += t * img_hist.at<float>(t);

  init_otsu_values();

  // End
  end = chrono::system_clock::now();
  elapsed_time += chrono::duration_cast<chrono::milliseconds>(end - start).count();

  cout << "Init algorithm values. All elapsed time is: " <<
    elapsed_time / 1000. << "seconds." << endl;
}

int otsu(int argc, char** argv) {
  cv::CommandLineParser parser(argc, argv, keys);

  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }
  
  const std::string filename = parser.get<std::string>(0);
  const std::string out_folder = parser.get<std::string>("out_folder");
  
  cv::Mat input_image = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  if (input_image.empty()) {
    std::cerr << "Could not open input image: " << filename
              << std::endl;
    return -1;
  }
  
  init_values(input_image);
  imshow("Original image.", img_gray);
  save_hist(out_folder);

  while (true) {

    if (flag_end) {
      cout << "The algorithm end work. Threshold value is "
        << threshold_value << endl;
      cout << "All elapsed time is: " <<
        elapsed_time / 1000. << "seconds." << endl;
    }

    char key = (char)cv::waitKey();
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
      case 'n':
      case 'N':
        std::cout << "Demonstration started || continued." << std::endl;
        calc_threshold();
        save_img(out_folder);
        continue;

      case 'r':
      case 'R':
      default:
        std::cout << "Demonstration restored." << std::endl;
        init_otsu_values();
        continue;
    }
  }
  
  return 0;
}
