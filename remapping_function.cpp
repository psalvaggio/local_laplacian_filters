// File Description
// Author: Philip Salvaggio

#include "remapping_function.h"

#include <cmath>
#include <algorithm>

using namespace std;

RemappingFunction::RemappingFunction(double alpha, double beta)
    : alpha_(alpha), beta_(beta) {}

RemappingFunction::~RemappingFunction() {}

double RemappingFunction::SmoothStep(double x_min, double x_max, double x) {
  double y = (x - x_min) / (x_max - x_min);
  y = max(0.0, min(1.0, y));
  return pow(y, 2) * pow(y-2, 2);
}
