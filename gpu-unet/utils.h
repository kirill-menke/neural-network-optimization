#pragma once

#include <string>
#include "./tensor.h"

void uniformRandomInit(float min, float max, Tensor<float, 4> &weights, Tensor<float, 1> &bias);
void heInit(Tensor<float, 4>& weights, Tensor<float, 1>& bias);

void writeToFile(const std::string &filename, const Tensor<float, 4> &weights, const Tensor<float, 1> &bias);
void readFromFile(const std::string &filename, Tensor<float, 4> &weights, Tensor<float, 1> &bias);

