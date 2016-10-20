///@File: ISegmentMotion.h
///@Brief: Contains interface for SegmentMotion classes
///@Author: Vitaliy Baldeev
///@Date: 10 October 2015

#pragma once

#include <string>

#include "opencv2\core\mat.hpp"

namespace cv 
{
    class VideoCapture;
}

///@class ISegmentMotion
/// Interface for SegmentMotion classes
class SegmentMotionBase
{
public:
    ///@brief Launch demonstration
    void Run();

        ///@brief Factory method
    static SegmentMotionBase* CreateAlgorithm(std::string& algorithmName);
    
public:
    ///@brief Destructor
    virtual ~SegmentMotionBase() {};
    
    ///@brief Get the name of algorithm 
    virtual std::string GetName() const 
    {
        return "SMBase";
    }

protected:
    ///@brief protected constructor
    SegmentMotionBase() {};

    ///@brief Create trackbars
    virtual void createGUI() {};

    ///@brief Apply the algorthm of background subtraction
    ///@return the result binary image
    virtual cv::Mat process(cv::VideoCapture& capture) 
    { 
        return cv::Mat();
    }

private:
    cv::Mat m_foreground;  ///< binary image with moving objects
};
