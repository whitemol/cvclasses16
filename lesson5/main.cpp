// lesson4.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "FeatureDetectorBase.h"
#include "FeatureDetectorDemo.h"

int main()
{
    std::cout << "Select the detector: \n"
        << "Harris  - Harris detector \n"
        << "FAST    - FAST detector \n"
        << "LoG     - LoG detector \n";

    std::string name;
    std::cin >> name;

    std::unique_ptr<FeatureDetectorDemo> pDemo;
    pDemo->Run(name);

    return 0;
}
