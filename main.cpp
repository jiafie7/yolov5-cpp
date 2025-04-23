#include "detector/YOLODetector.h"
#include <iostream>
#include <string>

using namespace detector;

void printUsage(const std::string& program_name)
{
  std::cout << "Available modes:" << std::endl;
  std::cout << "  --image <image_path>               Processing a single image" << std::endl;

  std::cout << std::endl;
  std::cout << "Example:" << std::endl;
  std::cout << "  " << program_name << " --image ../sample.jpg" << std::endl;
}

int main(int argc, char* argv[])
{
  try
  {
    std::string mode;
    if (argc > 1)
    {
      mode = argv[1];
    }
    else
    {
      printUsage(argv[0]);
      return 0;
    }

    // Creat detector instance 
    YOLODetector detector("../models/yolov5s.onnx", "../config/coco.names");

    if (mode == "--image" && argc > 2)
    {
      // Single image detection mode
      std::string image_path = argv[2];
      cv::Mat frame = cv::imread(image_path);
      if (frame.empty())
      {
        throw std::runtime_error("Unable to load image: " + image_path);
      }

      detector.detect(frame);
    }
    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }
}
