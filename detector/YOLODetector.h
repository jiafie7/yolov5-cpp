#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace detector
{
  class YOLODetector 
  {
  public:
    YOLODetector(
      const std::string& model_path,
      const std::string& classes_path,
      float input_width = 640.0,
      float input_height = 640.0,
      float confidence_threshold = 0.45,
      float score_threshold = 0.5,
      float nms_threshold = 0.45
    );

    virtual ~YOLODetector() = default;

    double getInferenceTime() const;
    float getInputWidth() const;
    float getInputHeight() const;

    // Single image object detection
    cv::Mat detect(const cv::Mat& image);

    // Batch process all images in a directory
    int detectBatch(const std::string& input_dir, const std::string& output_dir, bool show_progress = true);
    int detectBatch(const std::vector<std::string>& image_paths, const std::string& output_dir, bool show_progress = true);

    // Process video files
    int detectVideo(const std::string& video_path, const std::string& output_path, bool show_preview = true);

  private:
    
    // Load category names from file
    void loadClasses(const std::string& classes_path);

    // Image preprocessing
    std::vector<cv::Mat> preProcess(const cv::Mat& input_image);

    // Post-processing test results
    cv::Mat postProcess(cv::Mat& input_image, const std::vector<cv::Mat>& outputs);

    // Draw labels on the image
    void drawLabel(cv::Mat& input_image, const std::string& label, int left, int top);

    // Get all image files in a directory
    std::vector<std::string> getImageFiles(const std::string& directory);

  private:
    // model parameters
    float m_input_width;
    float m_input_height;
    float m_confidence_threshold;
    float m_score_threshold;
    float m_nms_threshold;

    // neural network
    cv::dnn::Net m_net;

    // category name
    std::vector<std::string> m_class_list;

    // inference time
    double m_inference_time;

    // font setup
    const float FONT_SCALE = 0.75;
    const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
    const int THICKNESS = 2;

    // color setup, cv::Scalar(B, G, R)
    const cv::Scalar BLACK = cv::Scalar(20, 20, 20);     
    const cv::Scalar SKY_BLUE = cv::Scalar(235, 206, 135);  
    const cv::Scalar ORANGE = cv::Scalar(60, 130, 255); 
    const cv::Scalar MINT_GREEN = cv::Scalar(170, 240, 180);    
  };
}
