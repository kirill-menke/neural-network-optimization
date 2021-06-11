#include <math.h>

#include "Loss.h"

float
CrossEntropyLoss::forward(
    std::shared_ptr<Eigen::Tensor<float, 2>> input_tensor,
    std::shared_ptr<Eigen::Tensor<float, 2>> label_tensor)
{
    this->input_tensor = input_tensor;

    const float EPSILON = 1e-9;
    auto dims = label_tensor->dimensions();
	int batcheSize = dims[0], features = dims[1];

    float loss = 0.;
    for (int b = 0; b < batcheSize; b++) {
        int maxIdx = 0;
        float max = 0; /* Ich glaub hier kann man 0 nehmen weil alle Werte nur von 0 bis 1? */
        for (int f = 0; f < features; f++) {
            float val = (*label_tensor)(b, f);
            if (val > max) {
                max = val;
                maxIdx = f;
            }
        }

        max = (*input_tensor)(b, maxIdx);
        loss += -log(max + EPSILON);
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

