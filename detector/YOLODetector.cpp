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
