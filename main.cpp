// File Description
// Author: Philip Salvaggio

#include "gaussian_pyramid.h"
#include "laplacian_pyramid.h"
#include "opencv_utils.h"
#include "remapping_function.h"

#include <iostream>
#include <sstream>

using namespace std;

void OutputBinaryImage(const std::string& filename, cv::Mat image) {
  FILE* f = fopen(filename.c_str(), "wb");
  for (int x = 0; x < image.cols; x++) {
    for (int y = 0; y < image.rows; y++) {
      double tmp = image.at<double>(y, x);
      fwrite(&tmp, sizeof(double), 1, f);
    }
  }
  fclose(f);
}

int main(int argc, char** argv) {
  const double kSigmaR = 0.1;
  const double kAlpha = 3;
  const double kBeta = 1;
  RemappingFunction r(kAlpha, kBeta);

  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " image_file" << endl;
    return 1;
  }

  cv::Mat input = cv::imread(argv[1]);
  if (input.data == NULL) {
    cerr << "Could not read input image." << endl;
    return 1;
  }
  imwrite("original.png", input);
  input.convertTo(input, CV_64F, 1 / 255.0);


  cout << "Input image: " << argv[1] << " Size: " << input.cols << " x "
       << input.rows << " Channels: " << input.channels() << endl;

  int num_levels = LaplacianPyramid::GetLevelCount(input.rows, input.cols, 30);
  cout << "Number of levels: " << num_levels << endl;

  const int kRows = input.rows;
  const int kCols = input.cols;

  vector<cv::Mat> channels;
  cv::split(input, channels);

  vector<cv::Mat> output_channels;
  for (size_t c = 0; c < channels.size(); c++) {
    GaussianPyramid gauss_input(channels[c], num_levels);

    LaplacianPyramid output(kRows, kCols, num_levels);
    gauss_input[num_levels].copyTo(output[num_levels]);

    for (int l = 0; l < num_levels; l++) {
      int subregion_size = 3 * ((1 << (l + 2)) - 1);
      int subregion_r = subregion_size / 2;
      cout << "Level " << (l+1) << " (" << output[l].rows << " x "
           << output[l].cols << "), footprint: " << subregion_size << "x"
           << subregion_size << " ..." << endl;

      for (int y = 0; y < output[l].rows; y++) {
        int full_res_y = (1 << l) * y;
        int roi_y0 = full_res_y - subregion_r;
        int roi_y1 = full_res_y + subregion_r + 1;
        cv::Range row_range(max(0, roi_y0), min(roi_y1, kRows));
        int full_res_roi_y = full_res_y - row_range.start;

        for (int x = 0; x < output[l].cols; x++) {
          double g0 = gauss_input(l, y, x);
          
          int full_res_x = (1 << l) * x;
          int roi_x0 = full_res_x - subregion_r;
          int roi_x1 = full_res_x + subregion_r + 1;
          cv::Range col_range(max(0, roi_x0), min(roi_x1, kCols));
          int full_res_roi_x = full_res_x - col_range.start;

          cv::Mat r0 = channels[c](row_range, col_range);
          cv::Mat remapped;
          r.Evaluate<double>(r0, remapped, g0, kSigmaR);

          LaplacianPyramid tmp_pyr(remapped, l + 1,
              {row_range.start, row_range.end - 1,
               col_range.start, col_range.end - 1});
          output(l, y, x) = tmp_pyr(l, full_res_roi_y >> l,
                                       full_res_roi_x >> l);
          if (l == 0 && y == output[l].rows - 1 && x == 150) {
            cout << "Subwindow is " << row_range.start << ", " 
                 << row_range.end - 1 << ", " << col_range.start
                 << ", " << col_range.end - 1 << endl;
            cout << "Trying to index at (" << (full_res_roi_y >> l) << ", "
                 << (full_res_roi_x >> l) << "), size is "
                 << tmp_pyr[l].size() << endl;
            cout << "Value is " << output(l, y, x) << endl;
            cout << "ROI:" << endl;
            cout << setprecision(6) << fixed;
            for (int y = 0; y < r0.rows; y++) {
              for (int x = 0; x < r0.cols; x++) {
                cout << r0.at<double>(y, x) << "\t";
              }
              cout << endl;
            }
            cout << "Remapped:" << endl;
            for (int y = 0; y < remapped.rows; y++) {
              for (int x = 0; x < remapped.cols; x++) {
                cout << remapped.at<double>(y, x) << "\t";
              }
              cout << endl;
            }

            cout << "Gaussian Pyramid:" << endl;
            GaussianPyramid g_pyr(remapped, l+1,
                {row_range.start, row_range.end - 1,
                 col_range.start, col_range.end - 1});
            for (int k = 0; k < l+2; k++) {
              cout << "Level " << k << endl;
              for (int y = 0; y < g_pyr[k].rows; y++) {
                for (int x = 0; x < g_pyr[k].cols; x++) {
                  cout << g_pyr(k, y, x) << "\t";
                }
                cout << endl;
              }
            }

            cout << "Laplcian Pyramid:" << endl;
            LaplacianPyramid pyr(remapped, l+1,
                {row_range.start, row_range.end - 1,
                 col_range.start, col_range.end - 1});
            for (int k = 0; k < l+2; k++) {
              cout << "Level " << k << endl;
              for (int y = 0; y < pyr[k].rows; y++) {
                for (int x = 0; x < pyr[k].cols; x++) {
                  cout.precision(6);
                  cout << pyr(k, y, x) << "\t";
                }
                cout << endl;
              }
            }
          }
        }
      }

      stringstream ss;
      ss << "level" << l << ".png";
      imwrite(ss.str(), ByteScale(output[l]));
      ss.str("");
      ss << "level" << l << ".bin";
      OutputBinaryImage(ss.str(), output[l]);
    }

    output_channels.push_back(output.Reconstruct());
    stringstream ss;
    ss << "reconstructed" << c << ".png";
    imwrite(ss.str(), ByteScale(output_channels.back()));
  }

  cv::Mat reconstructed;
  cv::merge(output_channels, reconstructed);
  reconstructed *= 255;
  reconstructed.convertTo(reconstructed, CV_8U);
  imwrite("reconstructed.png", reconstructed);

  return 0;
}
