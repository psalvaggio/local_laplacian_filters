// File Description
// Author: Philip Salvaggio

#ifndef LAPLACIAN_PYRAMID_H
#define LAPLACIAN_PYRAMID_H

#include <opencv2/opencv.hpp>

class LaplacianPyramid {
 public:
  explicit LaplacianPyramid(int rows, int cols, int num_levels);
  LaplacianPyramid(const cv::Mat& image, int num_levels);
  LaplacianPyramid(const cv::Mat& image, int num_levels,
                   const std::vector<int>& subwindow);

  LaplacianPyramid(LaplacianPyramid&& other);

  LaplacianPyramid(const LaplacianPyramid&) = delete;
  LaplacianPyramid& operator=(const LaplacianPyramid&) = delete;

  const cv::Mat& GetPyramidLevel(int level) const {
    return pyramid_.at(level);
  }
  const cv::Mat& operator[](int level) const { return pyramid_[level]; }
  cv::Mat& operator[](int level) { return pyramid_[level]; }
  cv::Mat& operator()(int level) { return pyramid_[level]; }
  double& operator()(int level, int row, int col) {
    return pyramid_[level].at<double>(row, col);
  }

  cv::Mat Reconstruct() const;

  static int GetLevelCount(int rows, int cols, int desired_base_size);

  friend std::ostream &operator<<(std::ostream &output,
                                  const LaplacianPyramid& pyramid);

 private:
  void CreatePyramid(const cv::Mat& image, int num_levels);

 private:
  std::vector<cv::Mat> pyramid_;
  std::vector<int> subwindow_;
};

#endif  // LAPLACIAN_PYRAMID_H
