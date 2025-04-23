#include "detector/YOLODetector.h"
#include <iostream>
#include <string>

using namespace detector;

void printUsage(const std::string& program_name)
{
  std::cout << "Available modes:" << std::endl;
  std::cout << "  --image <image_path>                  Process a single image" << std::endl;
  std::cout << "  --batch <input_dir> <output_dir>      Batch process images in a directory" << std::endl;
  std::cout << "  --video <video_path> <output_path>    Process a video file" << std::endl;
  std::cout << "  --camera <camera_id> <output_path>    Camera real-time detect (default ID: 0)" << std::endl;

  std::cout << std::endl;

  std::cout << "Example:" << std::endl;
  std::cout << "  " << program_name << " --image ../sample.jpg" << std::endl;
  std::cout << "  " << program_name << " --batch ../data ../results" << std::endl;
  std::cout << "  " << program_name << " --video ../video.mp4 ../xxx.mp4" << std::endl;
  std::cout << "  " << program_name << " --camera" << std::endl;
  std::cout << "  " << program_name << " --camera ../xxx.mp4" << std::endl;
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
    else if (mode == "--batch" && argc > 3) 
    {
      // Batch process mode
      std::string input_dir = argv[2];
      std::string output_dir = argv[3];
            
      int processed = detector.detectBatch(input_dir, output_dir);   
    }
    else if (mode == "--video" && argc > 3) 
    {
      // Video process mode
      std::string video_path = argv[2];
      std::string output_path = argv[3];
            
      detector.detectVideo(video_path, output_path);  
    }
    else if (mode == "--camera") 
    {
      // Camera mode
      int camera_id = 0;
            
      if (argc > 2) 
      {
        std::string output_path = argv[2];
        detector.detectCamera(camera_id, output_path);
      }
      else
      {
        detector.detectCamera(camera_id);
      }
    } 
    else 
    {
      // Mode error
      std::cerr << "Invalid mode or arguments not enough" << std::endl;
      printUsage(argv[0]);
      return 1;
    }

    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }
}
