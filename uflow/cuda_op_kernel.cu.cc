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


__global__ void CorrelationUpdate(
	  const int* rbot0_size,
	  const int* top_size,
	  const float* rbot0,
	  const float* rbot1,
	  float* top
	) {
  extern __shared__ char patch_data_char[];

  float *patch_data = (float *)patch_data_char;

  // First (upper left) position of kernel upper-left corner in current center position of neighborhood in image 1
  int x1 = blockIdx.x + 4;
  int y1 = blockIdx.y + 4;
  int item = blockIdx.z;
  int ch_off = threadIdx.x;

  // Load 3D patch into shared shared memory
  for (int j = 0; j < 1; j++) { // HEIGHT
    for (int i = 0; i < 1; i++) { // WIDTH
      int ji_off = (j + i) * rbot0_size[3];
      for (int ch = ch_off; ch < rbot0_size[3]; ch += 32) { // CHANNELS
	int idx1 = ((item * rbot0_size[1] + y1+j) * rbot0_size[2] + x1+i) * rbot0_size[3] + ch;
	int idxPatchData = ji_off + ch;
	patch_data[idxPatchData] = rbot0[idx1];
      }
    }
  }

  __syncthreads();

  __shared__ float sum[32];

  // Compute correlation
  for (int top_channel = 0; top_channel < top_size[1]; top_channel++) {
    sum[ch_off] = 0;

    int s2o = top_channel % 9 - 4;
    int s2p = top_channel / 9 - 4;

    for (int j = 0; j < 1; j++) { // HEIGHT
      for (int i = 0; i < 1; i++) { // WIDTH
	int ji_off = (j + i) * rbot0_size[3];
	for (int ch = ch_off; ch < rbot0_size[3]; ch += 32) { // CHANNELS
	  int x2 = x1 + s2o;
	  int y2 = y1 + s2p;

	  int idxPatchData = ji_off + ch;
	  int idx2 = ((item * rbot0_size[1] + y2+j) * rbot0_size[2] + x2+i) * rbot0_size[3] + ch;

	  sum[ch_off] += patch_data[idxPatchData] * rbot1[idx2];
	}
      }
    }

    __syncthreads();

    if (ch_off == 0) {
      float total_sum = 0;
      for (int idx = 0; idx < 32; idx++) {
	total_sum += sum[idx];
      }
      const int sumelems = rbot0_size[3];
      const int index = ((top_channel*top_size[2] + blockIdx.y)*top_size[3])+blockIdx.x;
      top[index + item*top_size[1]*top_size[2]*top_size[3]] = total_sum / (float)sumelems;
    }
  }
}

void CorrelationUpdateKernelLauncher(const int grid_dim0, const int grid_dim1, const int grid_dim2, const int* rbot0_size, const int* top_size, const float* rbot0, const float* rbot1, float* output) {
  TF_CHECK_OK(::tensorflow::GpuLaunchKernel(CorrelationUpdate, dim3(grid_dim0, grid_dim1, grid_dim2), dim3(32, 1, 1), 0, nullptr,
                                            rbot0_size, top_size, rbot0, rbot1, output));
}

#endif
