#include <iostream>
#include <memory>
#include <list>

#include "SegmentMotionMeanFilter.h"

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cerr << "Set name of input video" << std::endl;
    return -1;
  }

  SegmentMotionMeanFilter proc;
  proc.Run(argv[1]);

  return 0;
}
