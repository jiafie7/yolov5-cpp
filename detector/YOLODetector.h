#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace detector
{
  class YOLODetector 
  {
  public:
      
  private:
    /**
     * @brief Load category names from file
     * @param classes_path Category name file path
     */
    void loadClasses(const std::string& classes_path);

    /**
     * @brief Image preprocessing
     * @param input_image Input images
     * @return Network outputs
     */
    std::vector<cv::Mat> preProcess(const cv::Mat& input_image);

    /**
     * @brief Post-processing test results
     * @param input_image Input images
     * @param outputs Network outputs
     * @return Image with detection results
     */
    cv::Mat postProcess(cv::Mat& input_image, const std::vector<cv::Mat>& outputs);

    /**
     * @brief Draw labels on the image
     * @param input_image Input images
     * @param label Label texts
     * @param left Left border
     * @param top Top border
     */
    void drawLabel(cv::Mat& input_image, const std::string& label, int left, int top);

  priavte:
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
    const float FONT_SCALE = 1;
    const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
    const int THICKNESS = 2;

    // color setup, cv::Scalar(B, G, R)
    const cv::Scalar BLACK = cv::Scalar(20, 20, 20);     
    const cv::Scalar SKY_BLUE = cv::Scalar(235, 206, 135);  
    const cv::Scalar ORANGE = cv::Scalar(60, 130, 255); 
    const cv::Scalar MINT_GREEN = cv::Scalar(170, 240, 180);    
  };
}
