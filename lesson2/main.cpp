///@File: main.cpp
///@Brief: Contains implementation of entry point of the application
///@Author: Roman Golovanov
///@Date: 08 September 2015

#include "stdafx.h"

#include <stdio.h>
#include <string.h>
#include <map>
#include <functional>

#include "opencv2/highgui.hpp"

/// @brief Helper storage of all demo applications
class DemoRegistry {
public:
	/// @brief adds new entity into the registry
	void Add(const std::string& i_name, std::function<int(int, char**)> i_entry) {
		m_reg.push_back(std::make_pair(i_name, i_entry));
	}

	/// @brief prints all registerd demos
	void Print() const {
		int i = 0;
		printf("\n The list of registered demos:\n");
		for (const auto& item : m_reg) {
			printf("%d) %s\n", i++, item.first.data());
		}
	}

	/// @brief executes demo by index and pass arguments
	void Run(int i_id, int argc, char** argv) const {
		printf("Launching %s...\n", m_reg[i_id].first.data());
		m_reg[i_id].second(argc, argv);
	}
private:
	using Registry = std::vector<std::pair<std::string, std::function<int(int, char**)>>>;
	Registry m_reg;
};

// Declaration of Demo functions implemented in separate unit
int iterOptThreshold(int, char**);
int otsu(int, char**);
int gmm(int, char**);
int regionGrowing(int, char**) { return 0; /*TODO: implement in separate file*/ }
int splitAndMerge(int, char**);
int watershed(int, char**);		/*TODO: modify to interactive demo*/
int kmeans(int, char**);		/*TODO: modify to interactive demo for images*/
int gabortexture(int, char**) { return 0; /*TODO: implement in separate file*/ }
int graphcut(int, char**) { return 0; /*TODO: implement in separate file*/ }

///@brief Entry point
int _tmain(int argc, char** argv)
{
	DemoRegistry reg;
	reg.Add("Iterative Optimal Thresholding",	iterOptThreshold);
	reg.Add("Otsu Thresholding",				otsu);
	reg.Add("Gaussian Mixture Modelling",		gmm);
	reg.Add("Region Growing",					regionGrowing);
	reg.Add("Splitting and Merging",			splitAndMerge);
	reg.Add("Watershed by OpenCV",				watershed);
	reg.Add("K-means by OpenCV",				kmeans);
	reg.Add("Gabor Filter for Textures",		gabortexture);
	reg.Add("Graph Cut",						graphcut);

	printf("Welcome to Image Segmentation Demo\n"
		   "----------------------------------\n"
		   "Please enter ID of the Demo:\n");
	reg.Print();

	int id = 0;
	scanf_s("%d", &id);
	reg.Run(id, argc, argv);

	return 0;
}
