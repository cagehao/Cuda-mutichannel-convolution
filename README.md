# Cuda-mutichannel-convolution
 This is a realization of CUDA convolution, using global memory and shared memory
,require cuda11.5 and vs2017. It compares the cpu realization, cuda global memory and shared memory realization of mutichannel convolution with input image 3x1024x1024, kernel of 64x3x3x3(out x in x kernel_size x kernel_size). The code referenced to https://blog.csdn.net/qq_23301703/article/details/91045465, and modified to support mutiple channels.
