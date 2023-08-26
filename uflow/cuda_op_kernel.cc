/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Modified by Jennifer Caballero for research purposes.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("CorrelationRearrange")
    .Input("in_sizes: int32")
    .Input("out_sizes: int32")
    .Input("input: float")
    .Input("output_t: float")
    .Output("output: float");

void CorrelationRearrangeKernelLauncher(const int n, const int grid_dim1, const int grid_dim2, const int* in_sizes, const int* out_sizes, const float* in, float* out);

class CorrelationRearrangeOp : public OpKernel {
 public:
  explicit CorrelationRearrangeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(2);
    auto input = input_tensor.flat<float>();

    // Allocate a new output tensor
    const Tensor& o_tensor = context->input(3);
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, o_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template flat<float>();

    // Get the in/out tensor shapes
    const Tensor& in_shape_tensor = context->input(0);
    auto in_shape = in_shape_tensor.flat<int32>();
    const Tensor& out_shape_tensor = context->input(1);
    auto out_shape = out_shape_tensor.flat<int32>();

    // Compute n
    auto n = input_tensor.dim_size(2) * input_tensor.dim_size(3);

    // Call the cuda kernel launcher
    CorrelationRearrangeKernelLauncher(n, input_tensor.dim_size(1), input_tensor.dim_size(0), in_shape.data(), out_shape.data(), input.data(), output.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("CorrelationRearrange").Device(DEVICE_GPU), CorrelationRearrangeOp);


REGISTER_OP("CorrelationUpdate")
    .Input("rbot0_size: int32")
    .Input("top_size: int32")
    .Input("rbot0: float")
    .Input("rbot1: float")
    .Input("output_t: float")
    .Output("output: float");

void CorrelationUpdateKernelLauncher(const int grid_dim0, const int grid_dim1, const int grid_dim2, const int* rbot0_size, const int* top_size, const float* rbot0, const float* rbot1, float* output);


class CorrelationUpdateOp : public OpKernel {
 public:
  explicit CorrelationUpdateOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensors
    const Tensor& rbot0_tensor = context->input(2);
    auto rbot0 = rbot0_tensor.flat<float>();
    const Tensor& rbot1_tensor = context->input(3);
    auto rbot1 = rbot1_tensor.flat<float>();

    // Allocate a new output tensor
    const Tensor& o_tensor = context->input(4);
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, o_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template flat<float>();

    // Get the in/out tensor shapes
    const Tensor& rbot0_shape_tensor = context->input(0);
    auto rbot0_shape = rbot0_shape_tensor.flat<int32>();
    const Tensor& top_shape_tensor = context->input(1);
    auto top_shape = top_shape_tensor.flat<int32>();

    // Call the cuda kernel launcher
    CorrelationUpdateKernelLauncher(output_tensor->dim_size(3), output_tensor->dim_size(2), output_tensor->dim_size(0), rbot0_shape.data(), top_shape.data(), rbot0.data(), rbot1.data(), output.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("CorrelationUpdate").Device(DEVICE_GPU), CorrelationUpdateOp);
