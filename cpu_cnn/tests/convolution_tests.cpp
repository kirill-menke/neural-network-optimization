#include <iostream>
#include <assert.h>
#include <Eigen/Dense>

#include "../src/Conv.h"
#include "../src/utils.h"

void initializeInputs();

// 2D input
Eigen::Tensor<float, 4> input_image2d(1, 1, 14, 14);
Eigen::Tensor<float, 4> expected_result2d(1, 2, 14, 14);

Eigen::Tensor<float, 4> weights2d(2, 1, 3, 3);

Eigen::Tensor<float, 4> error_next2d(1, 2, 14, 14);
Eigen::Tensor<float, 4> expected_error_prev2d(1, 1, 14, 14);

Eigen::Tensor<float, 1> expected_gradient_bias2d(2);
Eigen::Tensor<float, 4> expected_gradient_weights2d(2, 1, 3, 3);


// 3D input
Eigen::Tensor<float, 4> input_image3d(1, 2, 14, 14);
Eigen::Tensor<float, 4> expected_result3d(1, 2, 14, 14);

Eigen::Tensor<float, 4> weights3d(2, 2, 3, 3);

Eigen::Tensor<float, 4> error_next3d(1, 2, 14, 14);
Eigen::Tensor<float, 4> expected_error_prev3d(1, 2, 14, 14);

Eigen::Tensor<float, 1> expected_gradient_bias3d(2);
Eigen::Tensor<float, 4> expected_gradient_weights3d(2, 2, 3, 3);


 void testConvForward2D(Eigen::Tensor<float, 1> bias, int times = 1) {
    Conv conv(2, 1, 3, 1);
    conv.setWeights(weights2d);
    conv.setBias(bias);

    auto exp_dims = expected_result2d.dimensions();

    // Adding bias to expected result
    Eigen::Tensor<float, 4> expected_result_bias(expected_result2d);
    for (int i = 0; i < exp_dims[0]; i++) {
        for (int j = 0; j < exp_dims[1]; j++) {
            for (int x = 0; x < exp_dims[2]; x++) {
                for (int y = 0; y < exp_dims[3]; y++) {
                    expected_result_bias(i, j, x, y) += bias(j);
                }
            }
        }
    }

    // Performing forward pass
    std::shared_ptr<Eigen::Tensor<float, 4>> output;
    for (int i = 0; i < times; i++) {
        output = conv.forward(std::make_shared<Eigen::Tensor<float, 4>>(input_image2d));
    }

    float diff = ((Eigen::Tensor<float, 0>)(expected_result_bias - *output).abs().sum())(0);
    
    assert(("FAILED", diff < 1e-7));
    std::cout << "PASSED" << std::endl;
 }


 void testConvForward3D(Eigen::Tensor<float, 1> bias, int times = 1) {
     Conv conv(2, 2, 3, 1);
     conv.setWeights(weights3d);
     conv.setBias(bias);

     auto exp_dims = expected_result3d.dimensions();

     // Adding bias to expected result
     Eigen::Tensor<float, 4> expected_result_bias(expected_result3d);
     for (int i = 0; i < exp_dims[0]; i++) {
         for (int j = 0; j < exp_dims[1]; j++) {
             for (int x = 0; x < exp_dims[2]; x++) {
                 for (int y = 0; y < exp_dims[3]; y++) {
                     expected_result_bias(i, j, x, y) += bias(j);
                 }
             }
         }
     }
     
     // Performing forward pass
     std::shared_ptr<Eigen::Tensor<float, 4>> output;
     for (int i = 0; i < times; i++) {
         output = conv.forward(std::make_shared<Eigen::Tensor<float, 4>>(input_image3d));
     }

     float diff = ((Eigen::Tensor<float, 0>)(expected_result_bias - *output).abs().sum())(0);

     assert(("FAILED", diff < 1e-7));
     std::cout << "PASSED" << std::endl;


 }


 void testConvBackward2D(int times = 1) {
     Conv conv(2, 1, 3, 1);
     conv.setWeights(weights2d);
     Eigen::Tensor<float, 1> bias(2);
     bias.setValues({ 0.5, 1 });
     conv.setBias(bias);

     
     // Performing backward pass
     std::shared_ptr<Eigen::Tensor<float, 4>> error_prev;
     for (int i = 0; i < times; i++) {
         conv.forward(std::make_shared<Eigen::Tensor<float, 4>>(input_image2d));
         error_prev = conv.backward(std::make_shared<Eigen::Tensor<float, 4>>(error_next2d));
     }

     float diff = ((Eigen::Tensor<float, 0>)(expected_error_prev2d - *error_prev).abs().sum())(0);
     assert(("Computation of error with respect to the previous layer is not correct.", diff < 1e-7));

     diff = ((Eigen::Tensor<float, 0>)(expected_gradient_bias2d - conv.getGradientBias()).abs().sum())(0);
     assert(("Computation of error with respect to the bias is not correct.", diff < 1e-7));

     diff = ((Eigen::Tensor<float, 0>)(expected_gradient_weights2d - conv.getGradientWeights()).abs().sum())(0);
     assert(("Computation of error with respect to the weights is not correct.", diff < 1e-7));

     std::cout << "PASSED" << std::endl;
 }


 void testConvBackward3D(int times = 1) {
     Conv conv(2, 2, 3, 1);
     conv.setWeights(weights3d);
     Eigen::Tensor<float, 1> bias(2);
     bias.setValues({ 0.5, 1.0 });
     conv.setBias(bias);


     // Performing backward pass
     std::shared_ptr<Eigen::Tensor<float, 4>> error_prev;
     for (int i = 0; i < times; i++) {
         conv.forward(std::make_shared<Eigen::Tensor<float, 4>>(input_image3d));
         error_prev = conv.backward(std::make_shared<Eigen::Tensor<float, 4>>(error_next3d));
     }

     float diff = ((Eigen::Tensor<float, 0>)(expected_error_prev3d - *error_prev).abs().sum())(0);
     assert(("Computation of error with respect to the previous layer is not correct.", diff < 1e-7));

     diff = ((Eigen::Tensor<float, 0>)(expected_gradient_bias3d - conv.getGradientBias()).abs().sum())(0);
     assert(("Computation of error with respect to the bias is not correct.", diff < 1e-7));

     diff = ((Eigen::Tensor<float, 0>)(expected_gradient_weights3d - conv.getGradientWeights()).abs().sum())(0);
     assert(("Computation of error with respect to the weights is not correct.", diff < 1e-7));

     std::cout << "PASSED" << std::endl;
 }


void initializeInputs() {
    input_image2d.setValues({ {{
     {1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. },
     {1. , 1. , 1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1. , 1. , 1.},
     {1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1.},
     {1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1.},
     {1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1.},
     {1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1.},
     {1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1.},
     {1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1.},
     {1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1.},
     {1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1.},
     {1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1.},
     {1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1.},
     {1. , 1. , 1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1. , 1. , 1.},
     {1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1.}}} });


    input_image3d.setValues({{
    {{1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. },
     {1. , 1. , 1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1. , 1. , 1.},
     {1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1.},
     {1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1.},
     {1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1.},
     {1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1.},
     {1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1.},
     {1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1.},
     {1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1.},
     {1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1.},
     {1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1.},
     {1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1.},
     {1. , 1. , 1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1. , 1. , 1.},
     {1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1.}},
        
     {{1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. },
     {1. , 1. , 1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1. , 1. , 1.},
     {1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1.},
     {1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1.},
     {1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1.},
     {1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1.},
     {1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1.},
     {1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1.},
     {1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1.},
     {1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1.},
     {1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1.},
     {1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1.},
     {1. , 1. , 1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1. , 1. , 1.},
     {1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1.}} }});


    expected_result2d.setValues({{
       {{ 1.5, 0., 0., 0., -0.25, -0.25, 0., 0., 0.25, 0.25, 0., 0., 0., -1.5 },
        {2., 0., -0.25, -0.25, -0.5, -0.5, 0., 0., 0.5, 0.5, 0.25, 0.25, 0., -2.},
        {2., -0.25, -0.75, -0.5, -0.25, -0.25, 0., 0., 0.25, 0.25, 0.5, 0.75, 0.25, -2.},
        {2., -0.75, -1., -0.25, 0., 0., 0., 0., 0., 0., 0.25, 1., 0.75, -2.},
        {1.75, -1., -0.75, 0., -0.25, -0.25, 0., 0., 0.25, 0.25, 0., 0.75, 1., -1.75},
        {1.25, -1., -0.25, 0., -0.75, -0.75, 0., 0., 0.75, 0.75, 0., 0.25, 1., -1.25},
        {1., -1., 0., 0., -1., -1., 0., 0., 1., 1., 0., 0., 1., -1.},
        {1., -1., 0., 0., -1., -1., 0., 0., 1., 1., 0., 0., 1., -1.},
        {1.25, -1., -0.25, 0., -0.75, -0.75, 0., 0., 0.75, 0.75, 0., 0.25, 1., -1.25},
        {1.75, -1., -0.75, 0., -0.25, -0.25, 0., 0., 0.25, 0.25, 0., 0.75, 1., -1.75},
        {2., -0.75, -1., -0.25, 0., 0., 0., 0., 0., 0., 0.25, 1., 0.75, -2.},
        {2., -0.25, -0.75, -0.5, -0.25, -0.25, 0., 0., 0.25, 0.25, 0.5, 0.75, 0.25, -2.},
        {2., 0., -0.25, -0.25, -0.5, -0.5, 0., 0., 0.5, 0.5, 0.25, 0.25, 0., -2.},
        {1.5, 0., 0., 0., -0.25, -0.25, 0., 0., 0.25, 0.25, 0., 0., 0., -1.5}},

       {{ 1.5, 2., 2., 2., 1.75, 1.25, 1., 1., 1.25, 1.75, 2., 2., 2., 1.5 },
        {0., 0., -0.25, -0.75, -1., -1., -1., -1., -1., -1., -0.75, -0.25, 0., 0.},
        {0., -0.25, -0.75, -1., -0.75, -0.25, 0., 0., -0.25, -0.75, -1., -0.75, -0.25, 0.},
        {0., -0.25, -0.5, -0.25, 0., 0., 0., 0., 0., 0., -0.25, -0.5, -0.25, 0.},
        {-0.25, -0.5, -0.25, 0., -0.25, -0.75, -1., -1., -0.75, -0.25, 0., -0.25, -0.5, -0.25},
        {-0.25, -0.5, -0.25, 0., -0.25, -0.75, -1., -1., -0.75, -0.25, 0., -0.25, -0.5, -0.25},
        {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
        {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
        {0.25, 0.5, 0.25, 0., 0.25, 0.75, 1., 1., 0.75, 0.25, 0., 0.25, 0.5, 0.25},
        {0.25, 0.5, 0.25, 0., 0.25, 0.75, 1., 1., 0.75, 0.25, 0., 0.25, 0.5, 0.25},
        {0., 0.25, 0.5, 0.25, 0., 0., 0., 0., 0., 0., 0.25, 0.5, 0.25, 0.},
        {0., 0.25, 0.75, 1., 0.75, 0.25, 0., 0., 0.25, 0.75, 1., 0.75, 0.25, 0.},
        {0., 0., 0.25, 0.75, 1., 1., 1., 1., 1., 1., 0.75, 0.25, 0., 0.},
        {-1.5, -2., -2., -2., -1.75, -1.25, -1., -1., -1.25, -1.75, -2., -2., -2., -1.5}}}});


    expected_result3d.setValues({
      {{{ 3., 0., 0., 0., -0.5, -0.5, 0., 0., 0.5, 0.5, 0., 0., 0., -3. },
        {4., 0., -0.5, -0.5, -1., -1., 0., 0., 1., 1., 0.5, 0.5, 0., -4.},
        {4., -0.5, -1.5, -1., -0.5, -0.5, 0., 0., 0.5, 0.5, 1., 1.5, 0.5, -4.},
        {4., -1.5, -2., -0.5, 0., 0., 0., 0., 0., 0., 0.5, 2., 1.5, -4.},
        {3.5, -2., -1.5, 0., -0.5, -0.5, 0., 0., 0.5, 0.5, 0., 1.5, 2., -3.5},
        {2.5, -2., -0.5, 0., -1.5, -1.5, 0., 0., 1.5, 1.5, 0., 0.5, 2., -2.5},
        {2., -2., 0., 0., -2., -2., 0., 0., 2., 2., 0., 0., 2., -2.},
        {2., -2., 0., 0., -2., -2., 0., 0., 2., 2., 0., 0., 2., -2.},
        {2.5, -2., -0.5, 0., -1.5, -1.5, 0., 0., 1.5, 1.5, 0., 0.5, 2., -2.5},
        {3.5, -2., -1.5, 0., -0.5, -0.5, 0., 0., 0.5, 0.5, 0., 1.5, 2., -3.5},
        {4., -1.5, -2., -0.5, 0., 0., 0., 0., 0., 0., 0.5, 2., 1.5, -4.},
        {4., -0.5, -1.5, -1., -0.5, -0.5, 0., 0., 0.5, 0.5, 1., 1.5, 0.5, -4.},
        {4., 0., -0.5, -0.5, -1., -1., 0., 0., 1., 1., 0.5, 0.5, 0., -4.},
        {3., 0., 0., 0., -0.5, -0.5, 0., 0., 0.5, 0.5, 0., 0., 0., -3.}},

        {{ 3., 4., 4., 4., 3.5, 2.5, 2., 2., 2.5, 3.5, 4., 4., 4., 3. },
        {0., 0., -0.5, -1.5, -2., -2., -2., -2., -2., -2., -1.5, -0.5, 0., 0.},
        {0., -0.5, -1.5, -2., -1.5, -0.5, 0., 0., -0.5, -1.5, -2., -1.5, -0.5, 0.},
        {0., -0.5, -1., -0.5, 0., 0., 0., 0., 0., 0., -0.5, -1., -0.5, 0.},
        {-0.5, -1., -0.5, 0., -0.5, -1.5, -2., -2., -1.5, -0.5, 0., -0.5, -1., -0.5},
        {-0.5, -1., -0.5, 0., -0.5, -1.5, -2., -2., -1.5, -0.5, 0., -0.5, -1., -0.5},
        {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
        {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
        {0.5, 1., 0.5, 0., 0.5, 1.5, 2., 2., 1.5, 0.5, 0., 0.5, 1., 0.5},
        {0.5, 1., 0.5, 0., 0.5, 1.5, 2., 2., 1.5, 0.5, 0., 0.5, 1., 0.5},
        {0., 0.5, 1., 0.5, 0., 0., 0., 0., 0., 0., 0.5, 1., 0.5, 0.},
        {0., 0.5, 1.5, 2., 1.5, 0.5, 0., 0., 0.5, 1.5, 2., 1.5, 0.5, 0.},
        {0., 0., 0.5, 1.5, 2., 2., 2., 2., 2., 2., 1.5, 0.5, 0., 0.},
        {-3., -4., -4., -4., -3.5, -2.5, -2., -2., -2.5, -3.5, -4., -4., -4., -3.}}}});


    weights2d.setValues({ {{{-0.5, 0, 0.5}, {-1, 0, 1}, {-0.5, 0, 0.5}}},
                        {{{-0.5, -1, -0.5}, {0, 0, 0}, {0.5, 1, 0.5}}} });

    weights3d.setValues({ {{{-0.5, 0, 0.5}, {-1, 0, 1}, {-0.5, 0, 0.5}}, {{-0.5, 0, 0.5}, {-1, 0, 1}, {-0.5, 0, 0.5}}},
                        {{{-0.5, -1, -0.5}, {0, 0, 0}, {0.5, 1, 0.5}}, {{-0.5, -1, -0.5}, {0, 0, 0}, {0.5, 1, 0.5}} } });


    error_next2d.setValues({{
    {{1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1.},
     {1. , 1. , 1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1. , 1. , 1.},
     {1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1.},
     {1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1.},
     {1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1.},
     {1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1.},
     {1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1.},
     {1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1.},
     {1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1.},
     {1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1.},
     {1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1.},
     {1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1.},
     {1. , 1. , 1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1. , 1. , 1.},
     {1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1.}},
        
    {{1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1.},
     {1. , 1. , 1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1. , 1. , 1.},
     {1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1.},
     {1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1.},
     {1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1.},
     {1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1.},
     {1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1.},
     {1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1.},
     {1. , 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 1.},
     {1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1.},
     {1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1.},
     {1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1.},
     {1. , 1. , 1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 1. , 1. , 1. , 1. , 1.},
     {1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 1.}}}});

    error_next2d = error_next2d * 0.25f;
    error_next2d.chip(0, 0).chip(1, 0) = error_next2d.chip(0, 0).chip(1, 0) * -1.f;

    error_next3d = error_next2d;


    expected_error_prev2d.setValues({{{
        { 0., 0.5, 0.5, 0.5, 0.5, 0.375, 0.25, 0.25, 0.25, 0.375, 0.5, 0.5, 0.5, 0.75 },
        {-0.5, 0., 0., -0.125, -0.125, -0.125, -0.25, -0.25, -0.375, -0.375, -0.25, -0.125, 0., 0.5},
        {-0.5, 0., 0., -0.125, -0.125, 0., 0., 0., -0.125, -0.25, -0.375, -0.375, -0.125, 0.5},
        {-0.5, 0.125, 0.125, 0., 0., 0., 0., 0., 0., 0., -0.125, -0.375, -0.25, 0.5},
        {-0.5, 0.125, 0.125, 0., 0., -0.125, -0.25, -0.25, -0.25, -0.125, 0., -0.25, -0.375, 0.375},
        {-0.375, 0.125, 0., 0., 0.125, 0., -0.25, -0.25, -0.375, -0.25, 0., -0.125, -0.375, 0.25},
        {-0.25, 0.25, 0., 0., 0.25, 0.25, 0., 0., -0.25, -0.25, 0., 0., -0.25, 0.25},
        {-0.25, 0.25, 0., 0., 0.25, 0.25, 0., 0., -0.25, -0.25, 0., 0., -0.25, 0.25},
        {-0.25, 0.375, 0.125, 0., 0.25, 0.375, 0.25, 0.25, 0., -0.125, 0., 0., -0.125, 0.375},
        {-0.375, 0.375, 0.25, 0., 0.125, 0.25, 0.25, 0.25, 0.125, 0., 0., -0.125, -0.125, 0.5},
        {-0.5, 0.25, 0.375, 0.125, 0., 0., 0., 0., 0., 0., 0., -0.125, -0.125, 0.5},
        {-0.5, 0.125, 0.375, 0.375, 0.25, 0.125, 0., 0., 0., 0.125, 0.125, 0., 0., 0.5},
        {-0.5, 0., 0.125, 0.25, 0.375, 0.375, 0.25, 0.25, 0.125, 0.125, 0.125, 0., 0., 0.5},
        {-0.75, -0.5, -0.5, -0.5, -0.375, -0.25, -0.25, -0.25, -0.375, -0.5, -0.5, -0.5, -0.5, 0.}}}});

    
    expected_error_prev3d.setValues(
        {{{{ 0., 0.5, 0.5, 0.5, 0.5, 0.375, 0.25, 0.25, 0.25, 0.375, 0.5, 0.5, 0.5, 0.75 },
        {-0.5, 0., 0., -0.125, -0.125, -0.125, -0.25, -0.25, -0.375, -0.375, -0.25, -0.125, 0., 0.5},
        {-0.5, 0., 0., -0.125, -0.125, 0., 0., 0., -0.125, -0.25, -0.375, -0.375, -0.125, 0.5},
        {-0.5, 0.125, 0.125, 0., 0., 0., 0., 0., 0., 0., -0.125, -0.375, -0.25, 0.5},
        {-0.5, 0.125, 0.125, 0., 0., -0.125, -0.25, -0.25, -0.25, -0.125, 0., -0.25, -0.375, 0.375},
        {-0.375, 0.125, 0., 0., 0.125, 0., -0.25, -0.25, -0.375, -0.25, 0., -0.125, -0.375, 0.25},
        {-0.25, 0.25, 0., 0., 0.25, 0.25, 0., 0., -0.25, -0.25, 0., 0., -0.25, 0.25},
        {-0.25, 0.25, 0., 0., 0.25, 0.25, 0., 0., -0.25, -0.25, 0., 0., -0.25, 0.25},
        {-0.25, 0.375, 0.125, 0., 0.25, 0.375, 0.25, 0.25, 0., -0.125, 0., 0., -0.125, 0.375},
        {-0.375, 0.375, 0.25, 0., 0.125, 0.25, 0.25, 0.25, 0.125, 0., 0., -0.125, -0.125, 0.5},
        {-0.5, 0.25, 0.375, 0.125, 0., 0., 0., 0., 0., 0., 0., -0.125, -0.125, 0.5},
        {-0.5, 0.125, 0.375, 0.375, 0.25, 0.125, 0., 0., 0., 0.125, 0.125, 0., 0., 0.5},
        {-0.5, 0., 0.125, 0.25, 0.375, 0.375, 0.25, 0.25, 0.125, 0.125, 0.125, 0., 0., 0.5},
        {-0.75, -0.5, -0.5, -0.5, -0.375, -0.25, -0.25, -0.25, -0.375, -0.5, -0.5, -0.5, -0.5, 0.}},

        {{ 0., 0.5, 0.5, 0.5, 0.5, 0.375, 0.25, 0.25, 0.25, 0.375, 0.5, 0.5, 0.5, 0.75 },
        {-0.5, 0., 0., -0.125, -0.125, -0.125, -0.25, -0.25, -0.375, -0.375, -0.25, -0.125, 0., 0.5},
        {-0.5, 0., 0., -0.125, -0.125, 0., 0., 0., -0.125, -0.25, -0.375, -0.375, -0.125, 0.5},
        {-0.5, 0.125, 0.125, 0., 0., 0., 0., 0., 0., 0., -0.125, -0.375, -0.25, 0.5},
        {-0.5, 0.125, 0.125, 0., 0., -0.125, -0.25, -0.25, -0.25, -0.125, 0., -0.25, -0.375, 0.375},
        {-0.375, 0.125, 0., 0., 0.125, 0., -0.25, -0.25, -0.375, -0.25, 0., -0.125, -0.375, 0.25},
        {-0.25, 0.25, 0., 0., 0.25, 0.25, 0., 0., -0.25, -0.25, 0., 0., -0.25, 0.25},
        {-0.25, 0.25, 0., 0., 0.25, 0.25, 0., 0., -0.25, -0.25, 0., 0., -0.25, 0.25},
        {-0.25, 0.375, 0.125, 0., 0.25, 0.375, 0.25, 0.25, 0., -0.125, 0., 0., -0.125, 0.375},
        {-0.375, 0.375, 0.25, 0., 0.125, 0.25, 0.25, 0.25, 0.125, 0., 0., -0.125, -0.125, 0.5},
        {-0.5, 0.25, 0.375, 0.125, 0., 0., 0., 0., 0., 0., 0., -0.125, -0.125, 0.5},
        {-0.5, 0.125, 0.375, 0.375, 0.25, 0.125, 0., 0., 0., 0.125, 0.125, 0., 0., 0.5},
        {-0.5, 0., 0.125, 0.25, 0.375, 0.375, 0.25, 0.25, 0.125, 0.125, 0.125, 0., 0., 0.5},
        {-0.75, -0.5, -0.5, -0.5, -0.375, -0.25, -0.25, -0.25, -0.375, -0.5, -0.5, -0.5, -0.5, 0.}}}}
    
    );


    expected_gradient_weights2d.setValues({
      {{{18.75, 22.5, 18.75},
        {22.5, 27., 22.5},
        {18.75, 22.5, 18.75}}},
      {{{-18.75, -22.5, -18.75},
        {-22.5, -27., -22.5},
        {-18.75, -22.5, -18.75}}}});


    expected_gradient_weights3d.setValues({{{
        { 18.75, 22.5, 18.75},
        {22.5, 27., 22.5},
        {18.75, 22.5, 18.75}},

        {{ 18.75, 22.5, 18.75},
        {22.5, 27., 22.5},
        {18.75, 22.5, 18.75}} },


        {{{-18.75, -22.5, -18.75},
        {-22.5, -27., -22.5},
        {-18.75, -22.5, -18.75}},

        {{-18.75, -22.5, -18.75},
        {-22.5, -27., -22.5},
        {-18.75, -22.5, -18.75}} } });


    expected_gradient_bias2d.setValues({ 33., -33. });
    expected_gradient_bias3d = expected_gradient_bias2d;
}


int main_(int argc, const char* argv[]) {
    initializeInputs();
    Eigen::Tensor<float, 1> zero_bias(2);
    Eigen::Tensor<float, 1> non_zero_bias(2);
    zero_bias.setValues({ 0., 0. });
    non_zero_bias.setValues({ 0.5, -0.5 });

    std::cout << "------ CONVOLUTION LAYER TESTS ------" << std::endl;
    std::cout << "2D Convolution forward without bias: ";
    testConvForward2D(zero_bias);

    std::cout << "2D Convolution forward with bias: ";
    testConvForward2D(non_zero_bias);

    std::cout << "3D Convolution forward without bias: ";
    testConvForward3D(zero_bias, 1);

    std::cout << "3D Convolution forward with bias: ";
    testConvForward3D(non_zero_bias, 1);

    std::cout << "Convolution forward multiple times: ";
    testConvForward3D(non_zero_bias, 3);

    std::cout << "2D Convolution backward one time: ";
    testConvBackward2D(1);

    std::cout << "3D Convolution backward one time: ";
    testConvBackward3D();

    std::cout << "Convolution backward multiple times: ";
    testConvBackward3D(3);

    return 0;
}