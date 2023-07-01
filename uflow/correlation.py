import cupy
#import cupy.cuda
import re
if hasattr(cupy, 'util'):
    import cupy.util as cupyutil
else:
    import cupy as cupyutil

import tensorflow as tf


class Stream:
    #ptr = torch.cuda.current_stream().cuda_stream
    #ptr = tf.raw_ops.StreamActive(stream_name="")
    # stream = tf.compat.v1.cuda.Stream()
    # ptr = stream.cuda_stream
    print('hello')


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

def cupy_kernel(strFunction, objectVariables):
    strKernel = globals()[strFunction]

    while True:
        objectMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

        if objectMatch is None:
            break

        intArg = int(objectMatch.group(2))

        strTensor = objectMatch.group(4)
        intSizes = tf.shape(objectVariables[strTensor])

        strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))

    while True:
        objectMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objectMatch is None:
            break

        intArgs = int(objectMatch.group(2))
        strArgs = objectMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(
            intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objectMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')

    return strKernel

def tf2cp(x):
    dlcapsule = tf.experimental.dlpack.to_dlpack(x)
    return cp.fromDlpack(dlcapsule)

#@cupyutil.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)

def cupy_run1(inp, out):
    cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
        'input': inp,
        'output': out
    }))(
        grid=tuple([int((n + 16 - 1) / 16), tf.shape(first)[1], tf.shape(first)[0]]),
        block=tuple([16, 1, 1]),
        args=[n, id(inp), id(out)],
        stream=Stream
    )

def cupy_run2(inp, out, n):
    inp_cp = tf2cp(inp)
    out_cp = tf2cp(out)
    cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
        'input': inp_cp,
        'output': out_cp
    }))(
        grid=tuple([int((n + 16 - 1) / 16), tf.shape(inp_cp)[1], tf.shape(inp_cp)[0]]),
        block=tuple([16, 1, 1]),
        args=[n, id(inp_cp), id(out_cp)],
        stream=Stream
    )

def cupy_run3(inp0, inp1, out):
    rbot0 = tf2cp(inp0)
    rbot1= tf2cp(inp1)
    output = tf2cp(out)
    cupy_launch('kernel_Correlation_updateOutput', cupy_kernel('kernel_Correlation_updateOutput', {
        'rbot0': rbot0,
        'rbot1': rbot1,
        'top': output
    }))(
        grid=tuple([tf.shape(output)[3], tf.shape(output)[2], tf.shape(output)[0]]),
        block=tuple([32, 1, 1]),
        shared_mem=tf.shape(first)[1] * 4,
        args=[n, id(rbot0), id(rbot1), id(output)],
        stream=Stream
    )

    
class FunctionCorrelation(tf.keras.layers.Layer):
    def __init__(self):
        super(FunctionCorrelation, self).__init__()

    def save_tensors(self, first, second, rbot0, rbot1):
        # Save tensors for backward computation
        tf.py_function(lambda: None, [first, second, rbot0, rbot1], [])


    def call(self, first, second):
        rbot0 = tf.zeros([tf.shape(first)[0], tf.shape(first)[2] + 8, tf.shape(first)[3] + 8, tf.shape(first)[1]], dtype=first.dtype)
        rbot0_tmp = tf.py_function(func=tf.experimental.dlpack.to_dlpack,
                             inp=[rbot0], Tout=first.dtype)
        rbot0_final = tf.py_function(func=cupy.fromDlpack, inp=[rbot0_tmp], Tout=first.dtype)
        rbot1 = tf.zeros([tf.shape(first)[0], tf.shape(first)[2] + 8, tf.shape(first)[3] + 8, tf.shape(first)[1]], dtype=first.dtype)

        first_tmp = tf.py_function(func=tf.experimental.dlpack.to_dlpack,
                             inp=[first], Tout=first.dtype)
        first_final = tf.py_function(func=cupy.fromDlpack, inp=[first_tmp], Tout=first.dtype)

        self.save_tensors(first, second, rbot0, rbot1)

        output = tf.zeros([tf.shape(first)[0], 81, tf.shape(first)[2], tf.shape(first)[3]], dtype=first.dtype)

        if tf.test.is_gpu_available():
            n = tf.shape(first)[2] * tf.shape(first)[3]
            #cupy run
            tf.py_function(func=cupy_run1, inp=[first_final, rbot0_final], Tout=rbot0_final.dtype)
            n = tf.shape(second)[2] * tf.shape(second)[3]
            #cupy run 2
            tf.py_function(func=cupy_run2, inp=[second, rbot1], Tout=rbot1.dtype)
            n = tf.shape(output)[1] * tf.shape(output)[2] * tf.shape(output)[3]

            #cupy run3
            tf.py_function(func=cupy_run3, inp=[rbot0, rbot1, output], Tout=rbot1.dtype)

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
                    cupy_launch('kernel_Correlation_updateGradFirst',
                                cupy_kernel('kernel_Correlation_updateGradFirst', {
                                    'rbot0': rbot0,
                                    'rbot1': rbot1,
                                    'gradOutput': gradOutput,
                                    'gradFirst': gradFirst,
                                    'gradSecond': None
                                }))(
                        grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                        block=tuple([512, 1, 1]),
                        args=[n, intSample, id(rbot0), id(rbot1), id(gradOutput),
                              id(gradFirst), None],
                        stream=Stream
                    )

            if gradSecond is not None:
                for intSample in range(tf.shape(first)[0]):
                    n = tf.shape(first)[1] * tf.shape(first)[2] * tf.shape(first)[3]
                    cupy_launch('kernel_Correlation_updateGradSecond',
                                cupy_kernel('kernel_Correlation_updateGradSecond', {
                                    'rbot0': rbot0,
                                    'rbot1': rbot1,
                                    'gradOutput': gradOutput,
                                    'gradFirst': None,
                                    'gradSecond': gradSecond
                                }))(
                        grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                        block=tuple([512, 1, 1]),
                        args=[n, intSample, id(rbot0), id(rbot1), id(gradOutput), None,
                              id(gradSecond)],
                        stream=Stream
                    )



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
        rbot0_tmp = tf.py_function(func=tf.experimental.dlpack.to_dlpack, inp=[rbot0], Tout=rbot0.dtype)
        rbot0_final = tf.py_function(func=cupy.fromDlpack, inp=[rbot0_tmp], Tout=rbot0.dtype)

        rbot1 = tf.zeros([tf.shape(second)[0], tf.shape(second)[2] + 8, tf.shape(second)[3] + 8, tf.shape(second)[1]], dtype=second.dtype)

        rbot1_tmp = tf.py_function(func=tf.experimental.dlpack.to_dlpack, inp=[rbot1], Tout=rbot1.dtype)

        rbot1_final = tf.py_function(func=cupy.fromDlpack, inp=[rbot1_tmp], Tout=rbot1.dtype)

        self.save_tensors(input, second, rbot0, rbot1)

        output = tf.zeros([tf.shape(second)[0], tf.shape(second)[1], tf.shape(second)[2], tf.shape(second)[3]], dtype=second.dtype)

        second_tmp = tf.py_function(func=tf.experimental.dlpack.to_dlpack, inp=[second], Tout=second.dtype)
        second_final = tf.py_function(func=cupy.fromDlpack, inp=[second_tmp], Tout=second.dtype)

        if tf.test.is_gpu_available():
            n = tf.shape(second)[2] * tf.shape(second)[3]
            # cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
            #     'input': second,
            #     'output': rbot1
            # }))(
            #     grid=tuple([int((n + 16 - 1) / 16), tf.shape(second)[1], tf.shape(second)[0]]),
            #     block=tuple([16, 1, 1]),
            #     args=[n, id(second), id(rbot1)],
            #     stream=Stream
            # )

            tf.py_function(func=cupy_run2, inp=[second_final, rbot1_final, n], Tout=rbot1_final.dtype)

            for intSample in range(tf.shape(second)[0]):
                n = tf.shape(second)[1] * tf.shape(second)[2] * tf.shape(second)[3]
                tf.py_function(func=cupy_run3, inp=[rbot0_final, rbot1_final, output], Tout=rbot1.dtype)
            #     cupy_launch('kernel_Correlation_updateGradFirst',
            #                 cupy_kernel('kernel_Correlation_updateGradFirst', {
            #                     'rbot0': rbot0,
            #                     'rbot1': rbot1,
            #                     'gradOutput': input,
            #                     'gradFirst': output,
            #                     'gradSecond': None
            #                 }))(
            #         grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
            #         block=tuple([512, 1, 1]),
            #         args=[n, intSample, id(rbot0), id(rbot1), id(input),
            #               id(output), None],
            #         stream=Stream
            #     )

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
                cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
                    'input': gradOutput,
                    'output': rbot0
                }))(
                    grid=tuple([int((n + 16 - 1) / 16), tf.shape(gradOutput)[1], tf.shape(gradOutput)[0]]),
                    block=tuple([16, 1, 1]),
                    args=[n, id(gradOutput), id(rbot0)],
                    stream=Stream
                )

            if gradInput is not None:
                n = tf.shape(gradInput)[1] * tf.shape(gradInput)[2] * tf.shape(gradInput)[3]
                cupy_launch('kernel_Correlation_updateOutput', cupy_kernel('kernel_Correlation_updateOutput', {
                    'rbot0': rbot0,
                    'rbot1': rbot1,
                    'top': gradInput
                }))(
                    grid=tuple([tf.shape(gradInput)[3], tf.shape(gradInput)[2], tf.shape(gradInput)[0]]),
                    block=tuple([32, 1, 1]),
                    shared_mem=tf.shape(gradOutput)[1] * 4,
                    args=[n, id(rbot0), id(rbot1), id(gradInput)],
                    stream=Stream
                )

            if gradSecond is not None:
                for intSample in range(tf.shape(second)[0]):
                    n = tf.shape(second)[1] * tf.shape(second)[2] * tf.shape(second)[3]
                    cupy_launch('kernel_Correlation_updateGradSecond',
                                cupy_kernel('kernel_Correlation_updateGradSecond', {
                                    'rbot0': rbot0,
                                    'rbot1': rbot1,
                                    'gradOutput': input,
                                    'gradFirst': None,
                                    'gradSecond': gradSecond
                                }))(
                        grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                        block=tuple([512, 1, 1]),
                        args=[n, intSample, id(rbot0), id(rbot1), id(input), None,
                              id(gradSecond)],
                        stream=Stream
                    )

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
