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

      cv::Mat result = detector.detect(frame);

      // show result
      cv::imshow("Output", result);
      std::cout << "Press any key to close the window..." << std::endl;
      cv::waitKey(0);

      // save result
      std::string output_path = image_path.substr(0, image_path.find_last_of('.')) + "_detected.jpg";
      cv::imwrite(output_path, result);
      std::cout << "Results saved to: " << output_path << std::endl;
    }
    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }
}
