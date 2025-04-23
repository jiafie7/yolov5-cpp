#include "YOLODetector.h"
#include <iostream>
#include <fstream>
#include <filesystem>

using namespace detector;

YOLODetector::YOLODetector(
  const std::string& model_path,
  const std::string& classes_path,
  float input_width,
  float input_height,
  float confidence_threshold,
  float score_threshold,
  float nms_threshold
)
  : m_input_width(input_width)
  , m_input_height(input_height)
  , m_confidence_threshold(confidence_threshold)
  , m_score_threshold(score_threshold)
  , m_nms_threshold(nms_threshold)
  , m_inference_time(0.0)
{
  // Load model
  m_net = cv::dnn::readNet(model_path);

  // Load category names
  loadClasses(classes_path);
}

double YOLODetector::getInferenceTime() const
{
  return m_inference_time;
}
    
float YOLODetector::getInputWidth() const
{
  return m_input_width;
}

float YOLODetector::getInputHeight() const
{
  return m_input_height;
}

// Load category names from file
void YOLODetector::loadClasses(const std::string& classes_path)
{
  std::ifstream ifs(classes_path);
  std::string line;
  while (std::getline(ifs, line))
    m_class_list.push_back(line);
}
   
// Image preprocessing
std::vector<cv::Mat> YOLODetector::preProcess(const cv::Mat& input_image)
{
  // Process input image
  cv::Mat blob;
  cv::dnn::blobFromImage(input_image, blob, 1.0 / 255.0, cv::Size(m_input_width, m_input_height), cv::Scalar(), true, false);

  // Set input to network
  m_net.setInput(blob);

  std::vector<cv::Mat> outputs;
  double start = static_cast<double>(cv::getTickCount());
  // Forward propagation
  m_net.forward(outputs, m_net.getUnconnectedOutLayersNames());
  double end = static_cast<double>(cv::getTickCount());

  // Compute inference time (ms)
  m_inference_time = (end - start) / cv::getTickFrequency() * 1000;

  return outputs;
}

// Post-processing test results
cv::Mat YOLODetector::postProcess(cv::Mat& input_image, const std::vector<cv::Mat>& outputs)
{
  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  // Scaling factor between original image size and model input size
  float x_factor = input_image.cols / m_input_width;
  float y_factor = input_image.rows / m_input_height;

  float* data = (float*)outputs[0].data;

  const int rows = outputs[0].size[1];
  const int dimensions = outputs[0].size[2];

  for (int i = 0; i < rows; ++ i)
  {
    // [x_center, y_center, width, height, confidence_score, class_scores...]
    float confidence = data[4];
    
    // Discard low confidence detection box
    if (confidence >= m_confidence_threshold)
    {
      float* classes_scores = data + 5;
      // Encapsulate the category scores into a cv::Mat
      cv::Mat scores(1, m_class_list.size(), CV_32FC1, classes_scores);

      // Get the maximum category score and its index
      cv::Point class_id;
      double max_class_score;
      cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

      // Discard low score detection box
      if (max_class_score > m_score_threshold)
      {
        // Save the information of the retain box
        confidences.push_back(confidence);
        class_ids.push_back(class_id.x);

        // Center coordinates 
        float cx = data[0];
        float cy = data[1];
        
        // The output detection box size of network
        float w = data[2];
        float h = data[3];

        // The coordinates and size of detection box are restored to the original image size
        int left = static_cast<int>((cx - 0.5 * w) * x_factor);
        int top = static_cast<int>((cy - 0.5 * h) * y_factor);
        int width = static_cast<int>(w * x_factor);
        int height = static_cast<int>(h * y_factor);

        // save detection boxes
        boxes.push_back(cv::Rect(left, top, width, height));
      }
    }
    // skip to the next column
    data += 85;
  }
  
  // Perform NMS
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, m_score_threshold, m_nms_threshold, indices);
  
  // Traversing the retain box
  for (size_t i = 0; i < indices.size(); ++ i) 
  {
    int idx = indices[i];
    cv::Rect box = boxes[idx]; 
  
    int left = box.x;
    int top = box.y;
    int width = box.width;
    int height = box.height;
        
    // Draw the detection box
    cv::rectangle(input_image, cv::Point(left, top), cv::Point(left + width, top + height), SKY_BLUE, 3 * THICKNESS);
  
    // Get the class name and its confidence label
    std::string label = cv::format("%.2f", confidences[idx]);
    label = m_class_list[class_ids[idx]] + ":" + label;

    // draw the class label
    drawLabel(input_image, label, left, top);
  }
    
  return input_image;
}

// Draw labels on the image
void YOLODetector::drawLabel(cv::Mat& input_image, const std::string& label, int left, int top)
{ 
  // Place the label on top of the bounding box
  int baseLine;
  cv::Size label_size = cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
  top = std::max(top, label_size.height);

  // Top left corner
  cv::Point tlc = cv::Point(left, top);
  // Bottom right corner
  cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseLine);

  // Draw black rectangle
  cv::rectangle(input_image, tlc, brc, BLACK, cv::FILLED);

  // Place a label on the black rectangle
  cv::putText(input_image, label, cv::Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, MINT_GREEN, THICKNESS);
}

// Single image object detection
cv::Mat YOLODetector::detect(const cv::Mat& image)
{
  cv::Mat frame = image.clone();

  std::vector<cv::Mat> detections = preProcess(frame);

  cv::Mat result = postProcess(frame, detections);

  // Add inference time label
  std::string label = cv::format("Inference time : %.2f ms", m_inference_time);
  cv::putText(result, label, cv::Point(20, 40), FONT_FACE, FONT_SCALE, ORANGE, THICKNESS);

  return result;
}

// Get all image files in a directory
std::vector<std::string> YOLODetector::getImageFiles(const std::string& directory) 
{
  std::vector<std::string> image_files;
    
  for (const auto& entry : std::filesystem::directory_iterator(directory)) 
  {
    if (entry.is_regular_file()) 
    {
      // Only handles files
      std::string extension = entry.path().extension().string();
      
      // The suffix names are unified into lowercase letters for comparison
      std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
      // Handles common image formats
      if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || 
        extension == ".bmp" || extension == ".tiff" || extension == ".tif") 
      {
        image_files.push_back(entry.path().string());
      }
    }
  }
    
  return image_files;
}

// Batch process all images in a directory
int YOLODetector::detectBatch(const std::string& input_dir, const std::string& output_dir, bool show_progress)
{
  std::vector<std::string> image_files = getImageFiles(input_dir);
    
  // Recursion call the overloaded method
  return detectBatch(image_files, output_dir, show_progress);
}

int YOLODetector::detectBatch(const std::vector<std::string>& image_paths, const std::string& output_dir, bool show_progress)
{
  if (!std::filesystem::exists(output_dir)) 
  {
    std::filesystem::create_directory(output_dir);
  }  

  int processed_count = 0;
  int total_files = image_paths.size();

  if (show_progress) 
  {
    std::cout << "Start processing " << total_files << " image files..." << std::endl;
  }

  for (size_t i = 0; i < total_files; ++ i) 
  {
    const std::string& image_path = image_paths[i];
        
    try 
    {
      // Load image
      cv::Mat frame = cv::imread(image_path);
      if (frame.empty()) 
      {
        if (show_progress) 
        {
          std::cerr << "Error: unable to load image: " << image_path << std::endl;
        }
        continue;
      }
            
      // Perform each single image detection
      cv::Mat result = detect(frame);
            
      // Get filename (remove path)
      std::filesystem::path path(image_path);
      std::string filename = path.filename().string();
            
      std::string output_path = output_dir + "/" + filename;
            
      // Save detection results
      cv::imwrite(output_path, result);
            
      ++ processed_count;
            
      // Show progress
      if (show_progress) 
      {
        std::cout << "[" << i + 1 << "/" << total_files << "] have processed: " << filename << ", inference time: " << m_inference_time << " ms" << std::endl;
      }
    } 
    catch (const std::exception& e) 
    {
      std::cerr << "Handle image " << image_path << " error: " << e.what() << std::endl;
    }
  }
  
  if (show_progress) 
  {
    std::cout << "Batch processing completed! Processed successfully " << processed_count << "/" << total_files << " images" << std::endl;
  }
    
  return processed_count;
}

// Process video files
int YOLODetector::detectVideo(const std::string& video_path, const std::string& output_path, bool show_preview)
{
  // Open video file
  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) 
  {
    throw std::runtime_error("Unable open video file: " + video_path);
  }

  // Get video attributes
  int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = cap.get(cv::CAP_PROP_FPS);
  int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

  // Create video writer
  cv::VideoWriter video_writer(
    output_path,
    cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
    fps,
    cv::Size(frame_width, frame_height)
  );

  if (!video_writer.isOpened()) 
  {
    throw std::runtime_error("Unable create output video file: " + output_path);
  }

  cv::Mat frame;
  int frame_count = 0;
    
  std::cout << "Start process video: " << video_path << std::endl;
  std::cout << "Resolution: " << frame_width << "x" << frame_height << ", FPS: " << fps << ", Total frames: " << total_frames << std::endl;

  // Process each frame
  while (cap.read(frame)) 
  {
    // Detect current frame
    cv::Mat result = detect(frame);
        
    // Write detection result to output video
    video_writer.write(result);
        
    ++ frame_count;
        
    // Show process progress
    if (frame_count % 10 == 0 || frame_count == total_frames) 
    {
      std::cout << "Process progress: " << frame_count << "/" << total_frames << " (" << static_cast<int>(100.0 * frame_count / total_frames) << "%)" << std::endl;
    }

    if (show_preview) 
    {
      cv::imshow("Video Processing", result);
            
      // Press ESC to quit
      if (cv::waitKey(1) == 27) 
      {
        std::cout << "User interrupt processing" << std::endl;
        break;
      }
    }
  }

  // Release resources
  cap.release();
  video_writer.release();
    
  if (show_preview) 
  {    
    cv::destroyAllWindows();
  }
    
  std::cout << "Video processing completed, processed " << frame_count << " frames，results saved to " << output_path << std::endl;
    
  return frame_count;
}

// Real-time camera detection
int YOLODetector::detectCamera(int camera_id, const std::string& output_path) 
{
  // Open camera
  cv::VideoCapture cap(camera_id);
  if (!cap.isOpened()) 
  {
    throw std::runtime_error("Unable open camera，ID: " + std::to_string(camera_id));
  }
    
  // Get video attributes
  int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = 30.0;

  // Create video writer
  cv::VideoWriter video_writer;
  if (!output_path.empty()) 
  {
    video_writer.open(
      output_path,
      cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
      fps,
      cv::Size(frame_width, frame_height)
    );
        
    if (!video_writer.isOpened()) 
    {
      throw std::runtime_error("Unable create output video file: " + output_path);
    }
  }

  cv::Mat frame;
  int frame_count = 0;
  double total_time = 0.0;
    
  std::cout << "Start camera real-time detection，press ESC to quit" << std::endl;

  while (true) 
  {
    // read each frame
    if (!cap.read(frame)) 
    {
      std::cerr << "Unable to read frames from camera" << std::endl;
      break;
    }
        
    // Record start time
    auto start_time = std::chrono::high_resolution_clock::now();
        
    // Perform detection
    cv::Mat result = detect(frame);
        
    // Record end time
    auto end_time = std::chrono::high_resolution_clock::now();

    // Compute the cost time of each detection
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
    total_time += duration;
    ++ frame_count;
        
    // Compute average FPS
    double avg_fps = frame_count / (total_time / 1000.0);
        
    // Show FPS on image
    std::string fps_text = cv::format("FPS: %.2f", avg_fps);
    cv::putText(result, fps_text, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.75, ORANGE, 2);
        
    // Write detection to video file 
    if (video_writer.isOpened()) 
    {
      video_writer.write(result);
    }
        
    // Show preview
    cv::imshow("Camera Detection", result);
        
    // Press ESC to quit
    if (cv::waitKey(1) == 27) 
    {
      std::cout << "User interrupt process" << std::endl;
      break;
    }
  }

  cap.release();
  if (video_writer.isOpened()) 
  {
    video_writer.release();
  }
  
  cv::destroyAllWindows();
    
  std::cout << "Camera detection is completed, a total of " << frame_count << " frames" << std::endl;
  std::cout << "Average FPS: " << (frame_count / (total_time / 1000.0)) << std::endl;
    
  if (!output_path.empty()) 
  {
    std::cout << "Results saved to " << output_path << std::endl;
  }
    
  return frame_count;
}
