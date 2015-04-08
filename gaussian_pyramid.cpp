// File Description
// Author: Philip Salvaggio

#include "gaussian_pyramid.h"
#include <iostream>

using namespace std;

GaussianPyramid::GaussianPyramid(const cv::Mat& image, int num_levels)
    : pyramid_(), subwindow_({0, image.rows - 1, 0, image.cols - 1}) {
  CreatePyramid(image, num_levels);
}

GaussianPyramid::GaussianPyramid(const cv::Mat& image, int num_levels,
                                 const vector<int>& subwindow)
    : pyramid_(), subwindow_(subwindow) {
  CreatePyramid(image, num_levels);
}

void GaussianPyramid::CreatePyramid(const cv::Mat& image, int num_levels) {
  pyramid_.reserve(num_levels + 1);
  pyramid_.emplace_back();
  image.convertTo(pyramid_.back(), CV_64F);

  if (image.cols >> num_levels == 0 || image.rows >> num_levels == 0) {
    cerr << "Warning: Too many levels requested. Image size " 
         << image.cols << " x " << image.rows << " and  " << num_levels 
         << " levels wer requested." << endl;
  }

  for (int l = 0; l < num_levels; l++) {
    const double a = 0.4;
    const cv::Mat& previous = pyramid_.back();

    std::vector<int> prev_subwindow, next_subwindow;
    GetLevelSize(pyramid_.size() - 1, &prev_subwindow);
    GetLevelSize(pyramid_.size(), &next_subwindow);

    const int next_rows = next_subwindow[1] - next_subwindow[0] + 1;
    const int next_cols = next_subwindow[3] - next_subwindow[2] + 1;
    int row_offset = ((prev_subwindow[0] % 2) == 0) ? 0 : 1;
    int col_offset = ((prev_subwindow[2] % 2) == 0) ? 0 : 1;

    pyramid_.emplace_back(next_rows, next_cols, previous.type());
    cv::Mat& next = pyramid_.back();

    const int kEndRow = row_offset + 2 * next_rows;
    const int kEndCol = col_offset + 2 * next_cols;
    for (int y = row_offset; y < kEndRow; y += 2) {
      for (int x = col_offset; x < kEndCol; x += 2) {
        double value = 0;
        double total_weight = 0;
      
        int row_start = std::max(0, y - 2);
        int row_end = std::min(previous.rows - 1, y + 2);
        for (int n = row_start; n <= row_end; n++) {
          double row_weight = WeightingFunction(n - y, a);

          int col_start = std::max(0, x - 2);
          int col_end = std::min(previous.cols - 1, x + 2);
          for (int m = col_start; m <= col_end; m++) {
            double weight = row_weight * WeightingFunction(m - x, a);
            total_weight += weight;
            value += weight * previous.at<double>(n, m);
          }
        }
        next.at<double>(y >> 1, x >> 1) = value / total_weight;
      }
    }
  }
}

GaussianPyramid::GaussianPyramid(GaussianPyramid&& other)
    : pyramid_(move(other.pyramid_)) {}

cv::Mat GaussianPyramid::Expand(int level, int times) const {
  if (times < 1) return pyramid_.at(level);
  times = min(times, level);

  cv::Mat base = pyramid_[level], expanded;

  for (int i = 0; i < times; i++) {
    std::vector<int> subwindow;
    GetLevelSize(level - i - 1, &subwindow);

    int out_rows = pyramid_[level - i - 1].rows;
    int out_cols = pyramid_[level - i - 1].cols;
    expanded.create(out_rows, out_cols, base.type());

    int row_offset = ((subwindow[0] % 2) == 0) ? 0 : 1;
    int col_offset = ((subwindow[2] % 2) == 0) ? 0 : 1;
    Expand(base, row_offset, col_offset, expanded);

    base = expanded;
  }

  return expanded;
}

ostream &operator<<(ostream &output, const GaussianPyramid& pyramid) {
  output << "Gaussian Pyramid:" << endl;
  for (size_t i = 0; i < pyramid.pyramid_.size(); i++) {
    output << "Level " << i << ": " << pyramid.pyramid_[i].cols << " x "
           << pyramid.pyramid_[i].rows;
    if (i != pyramid.pyramid_.size() - 1) output << endl;
  }
  return output;
}

void GaussianPyramid::GetLevelSize(int level, vector<int>* subwindow) const {
  GetLevelSize(subwindow_, level, subwindow);
}

void GaussianPyramid::GetLevelSize(const std::vector<int> base_subwindow,
                                   int level,
                                   std::vector<int>* subwindow) {
  subwindow->clear();
  subwindow->insert(begin(*subwindow),
      begin(base_subwindow), end(base_subwindow));

  for (int i = 0; i < level; i++) {
    (*subwindow)[0] = ((*subwindow)[0] >> 1) + (*subwindow)[0] % 2;
    (*subwindow)[1] = (*subwindow)[1] >> 1;
    (*subwindow)[2] = ((*subwindow)[2] >> 1) + (*subwindow)[2] % 2;
    (*subwindow)[3] = (*subwindow)[3] >> 1;
  }
}

inline double GaussianPyramid::WeightingFunction(int i, double a) {
  switch (i) {
    case 0: return a;
    case -1: case 1: return 0.25;
    case -2: case 2: return 0.25 - 0.5 * a;
  }
  return 0;
}

void GaussianPyramid::Expand(const cv::Mat& input,
                             int row_offset,
                             int col_offset,
                             cv::Mat& output) {
  const double a = 0.4;
  cv::Mat norm = cv::Mat::zeros(output.rows, output.cols, CV_64F);
  cv::Mat upsamp = cv::Mat::zeros(output.rows, output.cols, CV_64F);

  for (int i = row_offset; i < output.rows; i += 2) {
    for (int j = col_offset; j < output.cols; j += 2) {
      upsamp.at<double>(i, j) = input.at<double>(i >> 1, j >> 1);
      norm.at<double>(i, j) = 1;
    }
  }

  cv::Mat filter(5, 5, CV_64F);
  for (int i = -2; i <= 2; i++) {
    for (int j = -2; j <= 2; j++) {
      filter.at<double>(i + 2, j + 2) =
         WeightingFunction(i, a) * WeightingFunction(j, a);
    }
  }

  for (int i = 0; i < output.rows; i++) {
    int row_start = max(0, i - 2);
    int row_end = min(output.rows - 1, i + 2);
    for (int j = 0; j < output.cols; j++) {
      int col_start = max(0, j - 2);
      int col_end = min(output.cols - 1, j + 2);

      double value = 0;
      double total_weight = 0;
      for (int n = row_start; n <= row_end; n++) {
        for (int m = col_start; m <= col_end; m++) {
          double weight = filter.at<double>(n - i + 2, m - j + 2);
          value += weight * upsamp.at<double>(n, m);
          total_weight += weight * norm.at<double>(n, m);
        }
      }
      output.at<double>(i, j) = value / total_weight;
    }
  }
}
