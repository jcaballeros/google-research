/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"

__global__ void CorrelationRearrange(
		const int n,
		const int* in_sizes,
		const int* out_sizes,
		const float* input,
		float* output
	) {
          int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

	  if (intIndex >= n) {
	    return;
	  }

	  int intSample = blockIdx.z;
	  int intChannel = blockIdx.y;

	  float dblValue = input[(((intSample * in_sizes[1]) + intChannel) * in_sizes[2] * in_sizes[3]) + intIndex];

	  __syncthreads();

	  int intPaddedY = (intIndex / in_sizes[3]) + 4;
	  int intPaddedX = (intIndex % in_sizes[3]) + 4;
	  int intRearrange = ((in_sizes[3] + 8) * intPaddedY) + intPaddedX;

	  output[(((intSample * out_sizes[1] * out_sizes[2]) + intRearrange) * in_sizes[1]) + intChannel] = dblValue;

}

void CorrelationRearrangeKernelLauncher(const int n, const int grid_dim1, const int grid_dim2, const int* in_sizes, const int* out_sizes, const float* input, float* output) {
  int grid_dim0 = (int((n + 16 - 1) / 16));
  TF_CHECK_OK(::tensorflow::GpuLaunchKernel(CorrelationRearrange, dim3(grid_dim0, grid_dim1, grid_dim2), dim3(16, 1, 1), 0, nullptr,
                                            n, in_sizes, out_sizes, input, output));
}

#endif
