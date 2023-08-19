#Get location of Tensorflow headers and library files
#TF_INC=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#TF_LIB=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_INC='/usr/local/lib/python3.6/dist-packages/tensorflow/include/'
TF_LIB='/usr/local/lib/python3.6/dist-packages/tensorflow/'

CC        = gcc -O2 -pthread
CXX       = g++
GPUCC     = nvcc
CFLAGS    = -std=c++11 -I$(TF_INC) -D_GLIBCXX_USE_CXX11_ABI=0
GPUCFLAGS = -c
LFLAGS    =  -shared -fPIC -I$(TF_INC) -I$(TF_INC)/external/nsync/public -L$(TF_LIB) -L/usr/local/cuda/lib64 -I/usr/local/cuda/include /usr/local/lib/python3.6/dist-packages/tensorflow/libtensorflow_framework.so.2
GPULFLAGS = -x cu -shared -Xcompiler -fPIC -ltensorflow_framework -I$(TF_INC) -I$(TF_INC)/external/nsync/public -L$(TF_LIB)
DEBUG = -g -G
GPUDEF    = -D GOOGLE_CUDA=1
CGPUFLAGS = -lcudart


SRC       = cuda_op_kernel.cc
GPUSRC    = cuda_op_kernel.cu.cc
PROD      = cuda_op_kernel.so
GPUPROD = cuda_op_kernel_cu.so

default: gpu

cpu:
	$(CXX) $(CFLAGS) $(SRC) $(LFLAGS) -o $(PROD)

gpu:
	$(GPUCC) $(CFLAGS) $(GPUCFLAGS) $(GPUSRC) $(GPULFLAGS) -o $(GPUPROD) $(GPUDEF) -I/usr/local/ --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED
	$(CXX) $(CFLAGS)  $(SRC) $(GPUPROD) $(LFLAGS) $(CGPUFLAGS) -o $(PROD) $(GPUDEF)

clean:
	rm -f $(PROD) $(GPUPROD)
