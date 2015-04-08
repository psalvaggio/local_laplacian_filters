// File Description
// Author: Philip Salvaggio

#ifndef REMAPPING_FUNCTION_H
#define REMAPPING_FUNCTION_H

#include <opencv2/opencv.hpp>
#include <cmath>

class RemappingFunction {
 public:
  RemappingFunction(double alpha, double beta);
  ~RemappingFunction();

  double alpha() const { return alpha_; }
  void set_alpha(double alpha) { alpha_ = alpha; }

  double beta() const { return beta_; }
  void set_beta(double beta) { beta_ = beta; }

  template<typename T>
  T Evaluate(T value, double reference, double sigma_r);

  template<typename T>
  void Evaluate(const cv::Mat& input, cv::Mat& output,
      double reference, double sigma_r);

  double SmoothStep(double x_min, double x_max, double x);

 private:
  double alpha_, beta_;
};

template<typename T>
T RemappingFunction::Evaluate(T value, double reference, double sigma_r) {
  double delta = std::abs(value - reference);
  int sign = value < reference ? -1 : 1;

  double output = 0;
  if (delta < sigma_r) {
    double fraction = delta / sigma_r;
    double polynomial = pow(fraction, alpha_);
    if (alpha_ < 1) {
      const double kNoiseLevel = 0.01;
      double blend = SmoothStep(kNoiseLevel,
          2 * kNoiseLevel, fraction * sigma_r);
      polynomial = blend * polynomial + (1 - blend) * fraction;
    }
    output = reference + sign * sigma_r * polynomial;
  } else {
    double offset = (delta - sigma_r) * beta_;
    output = reference + sign * (sigma_r + offset);
  }
  return static_cast<T>(output);
}

template<typename T>
void RemappingFunction::Evaluate(const cv::Mat& input, cv::Mat& output,
      double reference, double sigma_r) {
  output.create(input.rows, input.cols, input.type());
  for (int i = 0; i < input.rows; i++) {
    for (int j = 0; j < input.cols; j++) {
      output.at<T>(i, j) =
          Evaluate(input.at<T>(i, j), reference, sigma_r);
    }
  }
}

#endif  // REMAPPING_FUNCTION_H
