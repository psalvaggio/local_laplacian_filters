// File Description
// Author: Philip Salvaggio

#ifndef GAUSSIAN_PYRAMID_H
#define GAUSSIAN_PYRAMID_H

#include <opencv2/opencv.hpp>
#include <iostream>

class GaussianPyramid {
 public:
  GaussianPyramid(const cv::Mat& image, int num_levels);
  GaussianPyramid(const cv::Mat& image, int num_levels,
                  const std::vector<int>& subwindow);

  GaussianPyramid(GaussianPyramid&& other);

  GaussianPyramid(const GaussianPyramid&) = delete;
  GaussianPyramid& operator=(const GaussianPyramid&) = delete;

  const cv::Mat& GetPyramidLevel(int level) const {
    return pyramid_.at(level);
  }
  const cv::Mat& operator[](int level) const { return GetPyramidLevel(level); }
  const double& operator()(int level, int row, int col) const {
    return pyramid_[level].at<double>(row, col);
  }
  double& operator()(int level, int row, int col) {
    return pyramid_[level].at<double>(row, col);
  }

  cv::Mat Expand(int level, int times) const;

  static void Expand(const cv::Mat& input,
                     int row_offset,
                     int col_offset,
                     cv::Mat& output);

  friend std::ostream &operator<<(std::ostream &output,
                                  const GaussianPyramid& pyramid);

  static void GetLevelSize(const std::vector<int> base_subwindow,
                           int level,
                           std::vector<int>* subwindow);
 private:
  void CreatePyramid(const cv::Mat& image, int num_levels);

  // i = -2, -1, 0, 1, 2
  // a = 0.3 - Broad blurring Kernel
  // s = 0.4   Gaussian-like kernel
  // a = 0.5 - Triangle
  // a = 0.6 - Trimodal (Negative lobes)
  static double WeightingFunction(int i, double a);

  void GetLevelSize(int level, std::vector<int>* subwindow) const;

 private:
  std::vector<cv::Mat> pyramid_;
  std::vector<int> subwindow_;
};

#endif  // GAUSSIAN_PYRAMID_H
