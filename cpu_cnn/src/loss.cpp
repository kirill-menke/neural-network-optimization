#include <math.h>

#include "loss.h"

float CrossEntropyLoss::forward(
    std::shared_ptr<Eigen::Tensor<float, 2>> input_tensor,
    std::shared_ptr<Eigen::Tensor<float, 2>> label_tensor)
{
    this->input_tensor = input_tensor;

    const float EPSILON = 1e-9f;
    auto dims = label_tensor->dimensions();
	int batcheSize = dims[0], features = dims[1];

    float loss = 0.;
    for (int b = 0; b < batcheSize; b++) {
        int idx = 0;
        for (int f = 0; f < features; f++) {
            float val = (*label_tensor)(b, f);
            if (val == 1.0) {
                idx = f;
                break;
            }
        }

        float pred = (*input_tensor)(b, idx);
        loss += -log(pred + EPSILON);
    }

    return loss;
}

std::shared_ptr<Eigen::Tensor<float, 2>>
CrossEntropyLoss::backward(std::shared_ptr<Eigen::Tensor<float, 2>> label_tensor)
{
    auto dims = label_tensor->dimensions();
	int batcheSize = dims[0], features = dims[1];

    auto output_tensor = std::make_shared<Eigen::Tensor<float, 2>>(batcheSize, features);

    for (int b = 0; b < batcheSize; b++) {
        for (int f = 0; f < features; f++) {
            (*output_tensor)(b, f) = -((*label_tensor)(b, f) / (*input_tensor)(b, f));
        }
    }

    return output_tensor;
}

