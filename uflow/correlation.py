import cupy
#import cupy.cuda
import re
if hasattr(cupy, 'util'):
    import cupy.util as cupyutil
else:
    import cupy as cupyutil

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

kernel_Correlation_rearrange = '''
	extern "C" __global__ void kernel_Correlation_rearrange(
		const int n,
		const float* input,
		float* output
	) {
	  int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

	  if (intIndex >= n) {
	    return;
	  }

	  int intSample = blockIdx.z;
	  int intChannel = blockIdx.y;

	  float dblValue = input[(((intSample * SIZE_1(input)) + intChannel) * SIZE_2(input) * SIZE_3(input)) + intIndex];

	  __syncthreads();

	  int intPaddedY = (intIndex / SIZE_3(input)) + 4;
	  int intPaddedX = (intIndex % SIZE_3(input)) + 4;
	  int intRearrange = ((SIZE_3(input) + 8) * intPaddedY) + intPaddedX;

	  output[(((intSample * SIZE_1(output) * SIZE_2(output)) + intRearrange) * SIZE_1(input)) + intChannel] = dblValue;
	}
'''

kernel_Correlation_updateOutput = '''
	extern "C" __global__ void kernel_Correlation_updateOutput(
	  const int n,
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
	      int ji_off = (j + i) * SIZE_3(rbot0);
	      for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) { // CHANNELS
	        int idx1 = ((item * SIZE_1(rbot0) + y1+j) * SIZE_2(rbot0) + x1+i) * SIZE_3(rbot0) + ch;
	        int idxPatchData = ji_off + ch;
	        patch_data[idxPatchData] = rbot0[idx1];
	      }
	    }
	  }

	  __syncthreads();

	  __shared__ float sum[32];

	  // Compute correlation
	  for (int top_channel = 0; top_channel < SIZE_1(top); top_channel++) {
	    sum[ch_off] = 0;

	    int s2o = top_channel % 9 - 4;
	    int s2p = top_channel / 9 - 4;

	    for (int j = 0; j < 1; j++) { // HEIGHT
	      for (int i = 0; i < 1; i++) { // WIDTH
	        int ji_off = (j + i) * SIZE_3(rbot0);
	        for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) { // CHANNELS
	          int x2 = x1 + s2o;
	          int y2 = y1 + s2p;

	          int idxPatchData = ji_off + ch;
	          int idx2 = ((item * SIZE_1(rbot0) + y2+j) * SIZE_2(rbot0) + x2+i) * SIZE_3(rbot0) + ch;

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
	      const int sumelems = SIZE_3(rbot0);
	      const int index = ((top_channel*SIZE_2(top) + blockIdx.y)*SIZE_3(top))+blockIdx.x;
	      top[index + item*SIZE_1(top)*SIZE_2(top)*SIZE_3(top)] = total_sum / (float)sumelems;
	    }
	  }
	}
'''

kernel_Correlation_updateGradFirst = '''
	#define ROUND_OFF 50000

	extern "C" __global__ void kernel_Correlation_updateGradFirst(
	  const int n,
	  const int intSample,
	  const float* rbot0,
	  const float* rbot1,
	  const float* gradOutput,
	  float* gradFirst,
	  float* gradSecond
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
	  int n = intIndex % SIZE_1(gradFirst); // channels
	  int l = (intIndex / SIZE_1(gradFirst)) % SIZE_3(gradFirst) + 4; // w-pos
	  int m = (intIndex / SIZE_1(gradFirst) / SIZE_3(gradFirst)) % SIZE_2(gradFirst) + 4; // h-pos

	  // round_off is a trick to enable integer division with ceil, even for negative numbers
	  // We use a large offset, for the inner part not to become negative.
	  const int round_off = ROUND_OFF;
	  const int round_off_s1 = round_off;

	  // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
	  int xmin = (l - 4 + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4)
	  int ymin = (m - 4 + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4)

	  // Same here:
	  int xmax = (l - 4 + round_off_s1) - round_off; // floor (l - 4)
	  int ymax = (m - 4 + round_off_s1) - round_off; // floor (m - 4)

	  float sum = 0;
	  if (xmax>=0 && ymax>=0 && (xmin<=SIZE_3(gradOutput)-1) && (ymin<=SIZE_2(gradOutput)-1)) {
	    xmin = max(0,xmin);
	    xmax = min(SIZE_3(gradOutput)-1,xmax);

	    ymin = max(0,ymin);
	    ymax = min(SIZE_2(gradOutput)-1,ymax);

	    for (int p = -4; p <= 4; p++) {
	      for (int o = -4; o <= 4; o++) {
	        // Get rbot1 data:
	        int s2o = o;
	        int s2p = p;
	        int idxbot1 = ((intSample * SIZE_1(rbot0) + (m+s2p)) * SIZE_2(rbot0) + (l+s2o)) * SIZE_3(rbot0) + n;
	        float bot1tmp = rbot1[idxbot1]; // rbot1[l+s2o,m+s2p,n]

	        // Index offset for gradOutput in following loops:
	        int op = (p+4) * 9 + (o+4); // index[o,p]
	        int idxopoffset = (intSample * SIZE_1(gradOutput) + op);

	        for (int y = ymin; y <= ymax; y++) {
	          for (int x = xmin; x <= xmax; x++) {
	            int idxgradOutput = (idxopoffset * SIZE_2(gradOutput) + y) * SIZE_3(gradOutput) + x; // gradOutput[x,y,o,p]
	            sum += gradOutput[idxgradOutput] * bot1tmp;
	          }
	        }
	      }
	    }
	  }
	  const int sumelems = SIZE_1(gradFirst);
	  const int bot0index = ((n * SIZE_2(gradFirst)) + (m-4)) * SIZE_3(gradFirst) + (l-4);
	  gradFirst[bot0index + intSample*SIZE_1(gradFirst)*SIZE_2(gradFirst)*SIZE_3(gradFirst)] = sum / (float)sumelems;
	} }
'''

kernel_Correlation_updateGradSecond = '''
	#define ROUND_OFF 50000

	extern "C" __global__ void kernel_Correlation_updateGradSecond(
	  const int n,
	  const int intSample,
	  const float* rbot0,
	  const float* rbot1,
	  const float* gradOutput,
	  float* gradFirst,
	  float* gradSecond
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
	  int n = intIndex % SIZE_1(gradSecond); // channels
	  int l = (intIndex / SIZE_1(gradSecond)) % SIZE_3(gradSecond) + 4; // w-pos
	  int m = (intIndex / SIZE_1(gradSecond) / SIZE_3(gradSecond)) % SIZE_2(gradSecond) + 4; // h-pos

	  // round_off is a trick to enable integer division with ceil, even for negative numbers
	  // We use a large offset, for the inner part not to become negative.
	  const int round_off = ROUND_OFF;
	  const int round_off_s1 = round_off;

	  float sum = 0;
	  for (int p = -4; p <= 4; p++) {
	    for (int o = -4; o <= 4; o++) {
	      int s2o = o;
	      int s2p = p;

	      //Get X,Y ranges and clamp
	      // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
	      int xmin = (l - 4 - s2o + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4 - s2o)
	      int ymin = (m - 4 - s2p + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4 - s2o)

	      // Same here:
	      int xmax = (l - 4 - s2o + round_off_s1) - round_off; // floor (l - 4 - s2o)
	      int ymax = (m - 4 - s2p + round_off_s1) - round_off; // floor (m - 4 - s2p)

	      if (xmax>=0 && ymax>=0 && (xmin<=SIZE_3(gradOutput)-1) && (ymin<=SIZE_2(gradOutput)-1)) {
	        xmin = max(0,xmin);
	        xmax = min(SIZE_3(gradOutput)-1,xmax);

	        ymin = max(0,ymin);
	        ymax = min(SIZE_2(gradOutput)-1,ymax);

	        // Get rbot0 data:
	        int idxbot0 = ((intSample * SIZE_1(rbot0) + (m-s2p)) * SIZE_2(rbot0) + (l-s2o)) * SIZE_3(rbot0) + n;
	        float bot0tmp = rbot0[idxbot0]; // rbot1[l+s2o,m+s2p,n]

	        // Index offset for gradOutput in following loops:
	        int op = (p+4) * 9 + (o+4); // index[o,p]
	        int idxopoffset = (intSample * SIZE_1(gradOutput) + op);

	        for (int y = ymin; y <= ymax; y++) {
	          for (int x = xmin; x <= xmax; x++) {
	            int idxgradOutput = (idxopoffset * SIZE_2(gradOutput) + y) * SIZE_3(gradOutput) + x; // gradOutput[x,y,o,p]
	            sum += gradOutput[idxgradOutput] * bot0tmp;
	          }
	        }
	      }
	    }
	  }
	  const int sumelems = SIZE_1(gradSecond);
	  const int bot1index = ((n * SIZE_2(gradSecond)) + (m-4)) * SIZE_3(gradSecond) + (l-4);
	  gradSecond[bot1index + intSample*SIZE_1(gradSecond)*SIZE_2(gradSecond)*SIZE_3(gradSecond)] = sum / (float)sumelems;
	} }
'''

class FunctionCorrelation(tf.keras.layers.Layer):
    def __init__(self):
        super(FunctionCorrelation, self).__init__()

    def call(self, first, second):
        rbot0 = tf.zeros([tf.shape(first)[0], tf.shape(first)[2] + 8, tf.shape(first)[3] + 8, tf.shape(first)[1]], dtype=first.dtype)
        rbot1 = tf.zeros([tf.shape(first)[0], tf.shape(first)[2] + 8, tf.shape(first)[3] + 8, tf.shape(first)[1]], dtype=first.dtype)

        output = tf.zeros([tf.shape(first)[0], 81, tf.shape(first)[2], tf.shape(first)[3]], dtype=first.dtype)

        if tf.test.is_gpu_available():
            n = tf.shape(first)[2] * tf.shape(first)[3]
            print('n value is ' + str(n.eval(session=tf.compat.v1.Session())))
            correlation_lib = tf.load_op_library('./uflow/cuda_op_kernel.so')
            rbot0 = correlation_lib.correlation_rearrange(tf.shape(first), tf.shape(rbot0), first, rbot0)
            rbot1 = correlation_lib.correlation_rearrange(tf.shape(second), tf.shape(rbot1), second, rbot1)

            output = correlation_lib.correlation_update(tf.shape(rbot0), tf.shape(output), rbot0, rbot1, output);

        elif not tf.test.is_gpu_available():
            raise NotImplementedError()

        return output

    def retrieve_tensors(*args):
        return args

    def backward(self, gradOutput):
        # Call the retrieve_tensors operation to get the saved tensors
        saved_tensors = tf.py_function(retrieve_tensors, [], [tf.float32, tf.float32, tf.float32, tf.float32])

        # Unpack the saved tensors
        first, second, rbot0, rbot1 = self.saved_tensors

        gradFirst = tf.zeros([tf.shape(first)[0], tf.shape(first)[1], tf.shape(first)[2], tf.shape(first)[3]], dtype=first.dtype) if \
        self.needs_input_grad[0] == True else None
        gradSecond = tf.zeros([tf.shape(first)[0], tf.shape(first)[1], tf.shape(first)[2], tf.shape(first)[3]], dtype=first.dtype) if \
        self.needs_input_grad[1] == True else None

        if tf.test.is_gpu_available():
        #if first.is_cuda == True:
            if gradFirst is not None:
                for intSample in range(tf.shape(first)[0]):
                    n = tf.shape(first)[1] * tf.shape(first)[2] * tf.shape(first)[3]

            if gradSecond is not None:
                for intSample in range(tf.shape(first)[0]):
                    n = tf.shape(first)[1] * tf.shape(first)[2] * tf.shape(first)[3]

        elif not tf.test.is_gpu_available():
            raise NotImplementedError()

        return gradFirst, gradSecond


class FunctionCorrelationTranspose(tf.keras.layers.Layer):
    def __init__(self):
        super(FunctionCorrelationTranspose, self).__init__()

    def save_tensors(self, input, second, rbot0, rbot1):
        # Save tensors for backward computation
        tf.py_function(lambda: None, [input, second, rbot0, rbot1], [])

    def call(self, input, second):
        rbot0 = tf.zeros([tf.shape(second)[0], tf.shape(second)[2] + 8, tf.shape(second)[3] + 8, tf.shape(second)[1]], dtype=second.dtype)
        rbot1 = tf.zeros([tf.shape(second)[0], tf.shape(second)[2] + 8, tf.shape(second)[3] + 8, tf.shape(second)[1]], dtype=second.dtype)
        output = tf.zeros([tf.shape(second)[0], tf.shape(second)[1], tf.shape(second)[2], tf.shape(second)[3]], dtype=second.dtype)

        if tf.test.is_gpu_available():
            n = tf.shape(second)[2] * tf.shape(second)[3]
            for intSample in range(tf.shape(second)[0]):
                n = tf.shape(second)[1] * tf.shape(second)[2] * tf.shape(second)[3]

        elif not tf.test.is_gpu_available():
            raise NotImplementedError()

        return output

    def retrieve_tensors(*args):
        return args

    def backward(self, gradOutput):
        # Call the retrieve_tensors operation to get the saved tensors
        saved_tensors = tf.py_function(retrieve_tensors, [], [tf.float32, tf.float32, tf.float32, tf.float32])

        # Unpack the saved tensors
        input, second, rbot0, rbot1 = self.saved_tensors

        gradInput = tf.zeros([tf.shape(input)[0], tf.shape(input)[1], tf.shape(input)(2), tf.shape(input)[3]], dtype=input.dtype) if \
        self.needs_input_grad[0] == True else None
        gradSecond = tf.zeros([tf.shape(second)[0], tf.shape(second)[1], tf.shape(second)[2], tf.shape(second)[3]], dtype=second.dtype) if \
        self.needs_input_grad[1] == True else None

        if tf.test.is_gpu_available():
            if gradInput is not None or gradSecond is not None:
                n = tf.shape(second)[2] * tf.shape(second)[3]

            if gradInput is not None:
                n = tf.shape(gradInput)[1] * tf.shape(gradInput)[2] * tf.shape(gradInput)[3]

            if gradSecond is not None:
                for intSample in range(tf.shape(second)[0]):
                    n = tf.shape(second)[1] * tf.shape(second)[2] * tf.shape(second)[3]

        elif not tf.test.is_gpu_available():
            raise NotImplementedError()

        return gradInput, gradSecond


# def FunctionCorrelation(reference_features, query_features):
#     return _FunctionCorrelation.apply(reference_features, query_features)

class ModuleCorrelation(tf.keras.layers.Layer):
    def __init__(self):
        super(ModuleCorrelation, self).__init__()

    def call(self, reference_features, query_features):
        return _FunctionCorrelation.apply(reference_features, query_features)


# def FunctionCorrelationTranspose(reference_features, query_features):
#     return _FunctionCorrelationTranspose.apply(reference_features, query_features)


class ModuleCorrelationTranspose(tf.keras.layers.Layer):
    def __init__(self):
        super(ModuleCorrelationTranspose, self).__init__()

    def call(self, reference_features, query_features):
        return _FunctionCorrelationTranspose.apply(reference_features, query_features)
