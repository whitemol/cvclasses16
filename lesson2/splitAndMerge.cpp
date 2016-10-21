#include "stdafx.h"

#include <iostream>
#include <stack>
#include <memory>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"


class QuadTree
{
public:
	QuadTree(cv::Mat in_image, const cv::Point& in_position, const cv::Size& in_size)
	{
		m_image = in_image;
		m_rectangle = cv::Rect(in_position, in_size);
		m_color = cv::Scalar(0, 0, 0);
	}

	std::shared_ptr<QuadTree> insert(int in_index)
	{
		switch (in_index)
		{
		case 0:
			m_child1 = std::make_shared<QuadTree>(m_image,
				m_rectangle.tl(),
				cv::Size((m_rectangle.width + 1) / 2,
				(m_rectangle.height + 1) / 2));
			return m_child1;
			break;

		case 1:
			m_child2 = std::make_shared<QuadTree>(m_image,
				m_rectangle.tl() + cv::Point((m_rectangle.width + 1) / 2, 0),
				cv::Size(m_rectangle.width - (m_rectangle.width + 1) / 2,
				(m_rectangle.height + 1) / 2));
			return m_child2;
			break;

		case 2:
			m_child3 = std::make_shared<QuadTree>(m_image,
				m_rectangle.tl() + cv::Point(0, (m_rectangle.height + 1) / 2),
				cv::Size((m_rectangle.width + 1) / 2,
					m_rectangle.height - (m_rectangle.height + 1) / 2));
			return m_child3;
			break;

		case 3:
			m_child4 = std::make_shared<QuadTree>(m_image,
				m_rectangle.tl() + cv::Point((m_rectangle.width + 1) / 2, (m_rectangle.height + 1) / 2),
				cv::Size(m_rectangle.width - (m_rectangle.width + 1) / 2,
					m_rectangle.height - (m_rectangle.height + 1) / 2));
			return m_child4;
			break;

		default:
			break;
		}
	}

	bool isHomogeneous()
	{
		bool first = true;
		cv::Vec3b value, temp;

		for (int y = m_rectangle.y; y < m_rectangle.y + m_rectangle.height; y++)
		{
			for (int x = m_rectangle.x; x < m_rectangle.x + m_rectangle.width; x++)
			{
				temp = m_image.at<cv::Vec3b>(cv::Point(x, y));

				if (first)
				{
					value = temp;
					first = false;
				}

				if (temp != value)
					return false;
			}
		}
		int total = int(temp[0]) + int(temp[1]) + int(temp[2]);
		if (total == 0)
			m_color = cv::Scalar(0, 255, 0);
		if (total == 255*3)
			m_color = cv::Scalar(255, 0, 0);

		return true;
	}

	cv::Scalar getColor() const
	{
		return m_color;
	}

	cv::Rect getRectangle() const
	{
		return m_rectangle;
	}

private:
	cv::Mat m_image;
	cv::Rect m_rectangle;
	std::shared_ptr<QuadTree> m_child1, m_child2, m_child3, m_child4;
	cv::Scalar m_color;
};


static void help()
{
	std::cout << "\nThis program demonstrates splitting and merging segmentation algorithm\n"
		"Usage:\n"
		"./splitAndMerge [image_name -- default is split.bmp]\n" << std::endl;
	std::cout << "Hot keys: \n"
		"\tSPACE - start split and merge algoritnm\n"
		"\tESC - quit the program\n"
		"\tr - restore the original image\n"
		"\tn - - next step\n";
}



int splitAndMerge(int argc, char** argv)
{
	cv::CommandLineParser parser(argc, argv, "{help h | | }{ @input | split.bmp | }");
	if (parser.has("help"))
	{
		help();
		return 0;
	}
	std::string filename = parser.get<std::string>("@input");
	cv::Mat original = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	if (original.empty())
	{
		std::cout << "Couldn't open image " << filename << ". Usage: splitAndMerge <image_name>\n";
		return 0;
	}
	help();
	
	cv::Mat image = original.clone();
	int width = image.size().width;
	int height = image.size().height;

	std::stack< std::shared_ptr<QuadTree> > leaves;
	std::shared_ptr<QuadTree> root;
	std::shared_ptr<QuadTree> current;

	cv::namedWindow("Split and merge demo");

	cv::Mat image_for_showing;
	cv::resize(image, image_for_showing, cv::Size(800, 800), 0.0, 0.0, CV_INTER_AREA);
	cv::putText(image_for_showing, "'Space' - start demonstration", cv::Point(0, 20),
		CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 0, 255));
	cv::putText(image_for_showing, "'Esc' - exit demonstration", cv::Point(0, 40),
		CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 0, 255));
	cv::imshow("Split and merge demo", image_for_showing);

	system("mkdir splitandmerge");
	bool cont = true;
	bool started = false;
	int64 start, end, iter_elapsed, total_elapsed = 0;

	int i = 0;
	while (cont)
	{
		int c = cv::waitKey(0);

		start = cvGetTickCount();

		switch (c)
		{
		case 32: // 'space', launch
			if (!started)
				started = true;
			else
				break;

		case 114: // 'r', restore
			start = cvGetTickCount();
			total_elapsed = 0;
			i = 0;
			image = original.clone();
			root.reset();
			root = std::make_shared<QuadTree>(image, cv::Point(0, 0), cv::Size(width, height));
			leaves.empty();
			leaves.push(root);	

		case 110: // 'n', next step
			if (!leaves.empty())
			{
				current = leaves.top();
				leaves.pop();
				if (!current->isHomogeneous())
				{
					leaves.push(current->insert(0));
					leaves.push(current->insert(1));
					leaves.push(current->insert(2));
					leaves.push(current->insert(3));
				}
				else
				{
					cv::Mat roi = image(current->getRectangle());
					roi.setTo(current->getColor());
				}
				i++;

				
				cv::resize(image, image_for_showing, cv::Size(800, 800), 0.0, 0.0, CV_INTER_AREA);

				end = cvGetTickCount();
				iter_elapsed = end - start;
				total_elapsed += iter_elapsed;


				std::string image_name = filename.substr(0, filename.find('.') - 1);
				std::string name = "splitandmerge/splitandmerge_" + image_name + "_" + std::to_string(i) + ".bmp";
				cv::imwrite(name.c_str(), image_for_showing);

				cv::putText(image_for_showing, "'N' - next step", cv::Point(0, 20),
					CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 0, 255));
				cv::putText(image_for_showing, "'R' - restore demonstration", cv::Point(0, 40),
					CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 0, 255));
				
				std::string text = "Step " + std::to_string(i);
				cv::putText(image_for_showing, text, cv::Point(0, 80),
					CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 0, 255));
				text = "iteration elapsed " + std::to_string(1000 * iter_elapsed / cv::getTickFrequency()) + " ms";
				cv::putText(image_for_showing, text, cv::Point(0, 100),
					CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 0, 255));
				text = "total elapsed " + std::to_string(1000 * total_elapsed / cv::getTickFrequency()) + " ms";
				cv::putText(image_for_showing, text, cv::Point(0, 120),
					CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 0, 255));

				cv::imshow("Split and merge demo", image_for_showing);
			}
			else
			{
				cont = false;
				cv::resize(image, image_for_showing, cv::Size(800, 800), 0.0, 0.0, CV_INTER_AREA);
				cv::putText(image_for_showing, "End of demonstration, press any key", cv::Point(0, 20),
			               CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0, 0, 255));
				cv::imshow("Split and merge demo", image_for_showing);
				cv::waitKey(0);
			}

			break;

		case 27: // 'esc', exit
		case -1: // // window closed
			cont = false;
			break;

		default:
			break;
		}
	}

	return 0;
}
