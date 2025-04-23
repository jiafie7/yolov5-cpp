#include "YOLODetector.h"
#include <iostream>
#include <fstream>

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
cv::Mat YOLODetector::postProcess(cv::Mat& input_image, const std::vector<cv::Mat>& outputs);

// Draw labels on the image
void YOLODetector::drawLabel(cv::Mat& input_image, const std::string& label, int left, int top)

// Single image object detection
cv::Mat YOLODetector::detect(const cv::Mat& image);
