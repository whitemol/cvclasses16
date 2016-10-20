///@File: gmm.cpp
///@Brief: Contains an example using the Gaussian Mixture Modelling
///@Author: Sergey Efremenkov
///@Date: 07 October 2016

#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <fstream>
#include <chrono>

#define PI 3.1415926535

static void help()
{
	std::cout << "\nThis program demonstrates the Gaussian Mixture Modelling\n"
		"Usage:\n"
		"./gmm [image_name -- default is ./gmm/gmm.png] [output_path --  default is ./gmm/]\n" << std::endl;
	std::cout << "Hot keys: \n"
		"\tESC - quit the program\n"
		"\tr - restore the original image\n"
		"\tw or SPACE - run Gaussian Mixture Modelling\n";
}

const cv::String keys =
			"{help h usage ? |                        | Help info     }"
			"{@input_image   | ./gmm/cameraman.png    | input image   }"
			"{@output_path   | ./gmm/                 | output path   }"
			;

cv::Mat grayImage, hist, histNorm, thresholdImage;

int numGaus, maxNumIter, iterNum;

double elapsedTime, eps;

std::vector<double> thresholds;

class parametersGaussian
{
	double mean;
	double variance;
	double weight;
public:
	parametersGaussian()
		: mean(0), variance(0), weight(0) {};

	parametersGaussian(double inMean, double inVariance, double inWeight)
		: mean(inMean), variance(inVariance), weight(inWeight) {};
	
	~parametersGaussian() {}

	double getMean()
	{
		return mean;
	}

	double getVar()
	{
		return variance;
	}

	double getWeight()
	{
		return weight;
	}
};

std::multimap<int, parametersGaussian> intermediate_data;

//Вычисление гистограммы
void histogram(cv::Mat inputImage)
{
	cv::cvtColor(inputImage, grayImage, CV_RGB2GRAY);

	int histSize = 256;

	float range[] = { 0, 256 };
	const float* histRange = { range };

	cv::calcHist(&grayImage, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
}

//Выводим гистограммы на экран и записываем её в файл
void showHistAndWrite(const std::string outputPath)
{
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / 256.0);

	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));

	cv::normalize(hist, histNorm, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

	//Формируем изображение гистограммы
	for (int i = 0; i < 256; ++i)
	{
		cv::rectangle(histImage, cv::Point(i*bin_w, hist_h), cv::Point((i + 1)*bin_w, hist_h - cvRound(histNorm.at<float>(i))), cvScalarAll(0), -1, 8, 0);
	}
	line(histImage, cv::Point(thresholds.back() * 2, hist_h), cv::Point(thresholds.back() * 2, 0), cv::Scalar(150, 200, 0), 2, 8, 0);

	imshow("Histogram of image", histImage);

	std::string fileName = outputPath + "gmm_" + "hist_" + std::to_string(iterNum) + ".png";

	try
	{
		cv::imwrite(fileName, histImage);
	}
	catch (...)
	{
		std::cout << "Histogram can not be saved" << fileName << std::endl;
	}
}

//Функция, которая формирует результат пороговой сегментации и выводит его на экран
void showResult()
{
	cv::threshold(grayImage, thresholdImage, thresholds.back(), 255, cv::THRESH_BINARY);
	cv::imshow("Image after thresholding", thresholdImage);
}

//Функция записывает гистограммы промежуточных результатов пороговой сегментации исходного изображения по указанному пути
void whriteAllIntermediateHist(const std::string outputPath)
{
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / 256.0);

	cv::normalize(hist, histNorm, 0, hist_h, cv::NORM_MINMAX, -1, cv::Mat());

	//Пробегаем по всем промежуточным порогам
	for (size_t i = 0; i < thresholds.size(); ++i)
	{
		cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(255, 255, 255));

		//Формируем изображение гистограммы
		for (int j = 0; j < 256; ++j)
		{
			cv::rectangle(histImage, cv::Point(j*bin_w, hist_h), cv::Point((j + 1)*bin_w, hist_h - cvRound(histNorm.at<float>(j))), cvScalarAll(0), -1, 8, 0);
		}
		line(histImage, cv::Point(thresholds[i] * 2, hist_h), cv::Point(thresholds[i] * 2, 0), cv::Scalar(150, 200, 0), 2, 8, 0);

		std::string fileName = outputPath + "gmm_" + "hist_" + std::to_string(i + 1) + "_" + std::to_string(iterNum) + ".png";

		try
		{
			cv::imwrite(fileName, histImage);
		}
		catch (...)
		{
			std::cout << "Histogram can not be saved" << fileName << std::endl;
		}
	}
}

//Функция записывает промежуточные результаты пороговой сегментации исходного изображения по указанному пути
void whriteAllIntermediateImages(const std::string outputPath)
{
	cv::Mat thresholdIm;

	//Пробегаем по всем промежуточным порогам
	for (size_t i = 0; i < thresholds.size(); ++i)
	{
		cv::threshold(grayImage, thresholdIm, thresholds[i], 255, cv::THRESH_BINARY);

		std::string fileName = outputPath + "gmm_ " + std::to_string(i + 1) + "_" + std::to_string(iterNum) + ".png";

		try
		{
			cv::imwrite(fileName, thresholdIm);
		}
		catch (...)
		{
			std::cout << "Image can not be saved" << fileName << std::endl;
		}
	}
}

//Функция записывает изображение по указанному пути
void whriteImage(const std::string outputPath)
{
	std::string fileName = outputPath + "gmm_ " + std::to_string(iterNum) + ".png";

	try
	{
		cv::imwrite(fileName, thresholdImage);
	}
	catch (...)
	{
		std::cout << "Image can not be saved"  << fileName << std::endl;
	}
}

//Функция для вычисления оптимального порогового значения через пересечение гауссиан
double thresholdCalc(double* means, double* variances, double* weights)
{
	double thresholdVal = 0;
	double A, B, C, discr;
	int indMax = 1;
	int indMin;

	if (means[0] > means[1])
	{
		indMax = 0;
	}
	indMin = (indMax + 1) % 2;

	A = variances[indMax] - variances[indMin];
	B = 2 * (means[indMax] * variances[indMin] - means[indMin] * variances[indMax]);
	C = means[indMin] * means[indMin] * variances[indMax] - means[indMax] * means[indMax] * variances[indMin]
		+ 2 * variances[indMax] * variances[indMin] * log((weights[indMax] * sqrt(variances[indMin])) / (weights[indMin] * sqrt(variances[indMax])));

	if ( A != 0)
	{
		double* calcThreshold = new double[2];

		discr = sqrt(B * B - 4 * A * C);

		calcThreshold[0] = (-B - discr) / (2 * A);
		calcThreshold[1] = (-B + discr) / (2 * A);
		for (int i = 0; i < 2; ++i)
		{
			if ((calcThreshold[i] > means[indMin]) && (calcThreshold[i] <= means[indMax]))
			{
				thresholdVal = calcThreshold[i];
				break;
			}
			else
			{
				if (calcThreshold[i] > 0)
				{
					thresholdVal = calcThreshold[i];
				}
			}
		}
		delete[] calcThreshold;
	}
	else
	{
		if (B != 0)
		{
			thresholdVal = -C / B;
		}
	}
	return thresholdVal;
}

void calcAllThresholds()
{
	double* meansGaus = new double[numGaus];
	double* varsGaus = new double[numGaus];
	double* weightsGaus = new double[numGaus];
	
	int ind[2] = { 1, 0 };

	if (intermediate_data.count(1) > intermediate_data.count(0))
	{
		ind[0] = 0;
		ind[1] = 1;
	};

	std::pair <std::multimap<int, parametersGaussian>::iterator, std::multimap<int, parametersGaussian>::iterator> ret[2];
	std::multimap<int, parametersGaussian>::iterator it[2];

	for (int i = 0; i < 2; ++i)
	{
		ret[i] = intermediate_data.equal_range(ind[i]);
		it[i] = ret[i].first;
	}

	for (; it[1] != ret[1].second; ++it[1], ++it[0])
	{
		if (it[0] == ret[0].second)
		{
			--it[0];
		}

		for (int i = 0; i < 2; ++i)
		{
			meansGaus[i] = it[i]->second.getMean();
			varsGaus[i] = it[i]->second.getVar();
			weightsGaus[i] = it[i]->second.getWeight();
		}
		thresholds.push_back(255*thresholdCalc(meansGaus, varsGaus, weightsGaus));
	}

	delete[] weightsGaus;
	delete[] varsGaus;
	delete[] meansGaus;
}

void EMAlg()
{
	double thresholdVal;

	std::vector<double> weights(numGaus);
	std::vector<double> variances(numGaus);
	std::vector<double> means(numGaus);
	std::vector<double> w(256);

	double elemHist;
	double numerW, denominW;
	double koeff, degree;
	double add;
	double sumForWeights, sumForMeans, sumForVar;
	double likelihood, newLikelihood;
	int numIter;

	bool converged;

	//Задаем начальные параметры гауссианов и весовых коэффициентов

	for (int i = 0; i < numGaus; ++i)
	{
		weights[i] = 1.0 / numGaus;
		variances[i] = 0.1;
		means[i] = (rand() % 100) / (100 * 1.0);
	}

	//Перебираем гауссианы
	for (int j = 0; j < numGaus; ++j)
	{
		likelihood = 0;
		numIter = 0;
		converged = false;
		//Решаем задачу минимизации - СКО реальной гистограммы и синтезированной должны быть минимальны
		while (!converged)
		{
			sumForWeights = 0;
			sumForMeans = 0;
			newLikelihood = 0;
			thresholdVal = 0;
			//Пробегаемся по всему диапазону гистограммы для вычисления плотности распределения вероятностей, нового математического ожидания и весового коэффициента
			for (int i = 0; i < 256; ++i)
			{
				elemHist = hist.at<float>(i) / grayImage.total();
				denominW = 0;

				for (int l = 0; l < numGaus; ++l)
				{
					koeff = 1 / (sqrt(2 * PI * variances[l]));
					degree = -((elemHist - means[l]) * (elemHist - means[l])) / (2 * variances[l]);
					add = koeff * exp(degree);
					denominW += weights[l] * add;
					if (l == j)
					{
						numerW = weights[l] * add;
						newLikelihood += log(add);
					}
				}

				w[i] = numerW / denominW;
				sumForWeights += w[i];
				sumForMeans += w[i] * elemHist;
			}

			intermediate_data.insert(std::pair<int, parametersGaussian>(j, parametersGaussian(means[j], variances[j], weights[j])));

			//Проверяем выполнились ли критерии остановки EM алгоритма: 
			//изменение логарифмического правдоподобия  стало меньше, чем ε, или достигнуто максимальное число итераций. 
			if ((abs(likelihood - newLikelihood) >= eps) && (numIter != maxNumIter))
			{
				weights[j] = sumForWeights / 256.0;
				means[j] = sumForMeans / (256.0 * weights[j]);

				sumForVar = 0;
				// Вычисляем новое значение дисперсии
				for (int i = 0; i < 256; ++i)
				{
					elemHist = hist.at<float>(i) / grayImage.total();
					sumForVar += w[i] * (elemHist - means[j]) * (elemHist - means[j]);
				}
				variances[j] = sumForVar / (256.0 * weights[j]);
			}
			else
			{
				converged = true;
			}
			numIter++;
			likelihood = newLikelihood;
		}
	}
	means.clear();
	variances.clear();
	weights.clear();
}

//Задаем начальные параметры для алгоритма gmm
void initParam()
{
	iterNum = 1;
	maxNumIter = 100;
	eps = 0.01;
	elapsedTime = 0;
	numGaus = 2;
}

int gmm(int argc, char** argv)
{
	std::chrono::time_point<std::chrono::system_clock> start, end;
	cv::CommandLineParser parser(argc, argv, keys);
	if (parser.has("help"))
	{
		help();
		parser.printMessage();
		return 0;
	}

	const std::string filename = parser.get<std::string>("@input_image");
	const std::string outputPath = parser.get<std::string>("@output_path");

	cv::Mat inputImage = cv::imread(filename, 1);
	if (inputImage.empty())
	{
		std::cout << "Could not open image " << filename << ". Usage: gmm <image_name>\n";
		return 0;
	}

	histogram(inputImage);
	initParam();

	imshow("original image", inputImage);
	imshow("gray image", grayImage);

	while (true)
	{
		thresholds.clear();

		intermediate_data.clear();

		char key = static_cast<char>(cv::waitKey());

		switch (key)
		{
		case 27:
		case 'q':
		case 'Q':
			std::cout << "Exit from program." << std::endl;
			return 0;
		case ' ':
		case 'L':
		case 'l':
			std::cout << "\nDemonstration started." << std::endl;
			start = std::chrono::system_clock::now();
			EMAlg();
			calcAllThresholds();
			whriteAllIntermediateHist(outputPath);
			whriteAllIntermediateImages(outputPath);
			showResult();
			whriteImage(outputPath);
			showHistAndWrite(outputPath);
			end = std::chrono::system_clock::now();
			elapsedTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			std::cout << "Threshold value: " << thresholds.back() << std::endl;
			std::cout << "Number of Gaussian mixture: " << numGaus << std::endl;
			std::cout << "Current elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms." << std::endl;
			std::cout << "Total elapsed time: " << elapsedTime << std::endl;
			iterNum++;
			continue;
		case 'r':
		case 'R':
			std::cout << "\nDemonstration restored." << std::endl;
			initParam();
			continue;
		}
	}

	return 0;
}