cmake_minimum_required(VERSION 2.8)
add_definitions(-std=c++11)

project( lesson1 )

find_package( OpenCV REQUIRED )
file(GLOB lesson1_SRC
	"*.h"
	"*.cpp"
)
add_executable(lesson1 ${lesson1_SRC})

# OpenCV LIBS
LINK_DIRECTORIES(/usr/local/lib)
TARGET_LINK_LIBRARIES(lesson1 opencv_calib3d opencv_contrib opencv_core opencv_features2d opencv_flann opencv_gpu opencv_highgui opencv_imgproc opencv_legacy opencv_ml opencv_objdetect opencv_ocl opencv_photo opencv_stitching opencv_superres opencv_ts -lopencv_video opencv_videostab opencv_videoio)

# OpenCV headers
include_directories(/usr/local/include/opencv)
include_directories(/usr/local/include/opencv2)
