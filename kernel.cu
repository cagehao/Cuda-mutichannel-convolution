#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include"device_launch_parameters.h"
#include"cuda_runtime.h"
#include<device_functions.h>
#include<windows.h >


#define MASK_WIDTH 3
int filter_size = MASK_WIDTH;
int arr_size = 1024;
int res_size = arr_size;
#define O_TILE_WIDTH 32
#define BLOCK_WIDTH (O_TILE_WIDTH + MASK_WIDTH - 1)
#define IN_CHANNELS 3
using namespace std;
#define checkCUDNN(expression)                                  \
  {                                                             \
    cudnnStatus_t status = (expression);                        \
    if (status != CUDNN_STATUS_SUCCESS) {                       \
	    printf(cudnnGetErrorString(status));                 \
    }                                                           \
 }

//cpu
double Conv2(double**** filter, double*** arr, double*** res, int filter_size, int arr_size, int C, int K, int P) {
	double temp;
	double sum = 0.0;
	for (int k = 0; k < K; k++) {
		for (int i = P; i < arr_size -P; i++) {
			for (int j = P; j < arr_size -P; j++) {

				temp = 0;
				int starti = i -P;
				int startj = j - P;
				for (int m = 0; m <  filter_size; m++) {
					for (int n = 0; n <  filter_size; n++) {
						for (int c = 0; c < C; c++) {
							temp += filter[m][n][c][k] * arr[m + starti][n + startj][c];
						}


					}
				}
				//res[i-P][j-P][k] = temp;
				sum += temp;
			}
		}
	}
	return sum;
}

//global memory gpu version
__global__
void convolution_2D_basic(double *in, double *out, double *mask, int maskwidth,int w, int h, int C, int K, int P, int pad_W, int pad_H) {
	int Col = blockIdx.x*blockDim.x + threadIdx.x+P;
	int Row = blockIdx.y*blockDim.y + threadIdx.y+P;
	if (Row < h+P&&Col < w+P) {
		
		//start
		int startCol = Col - P;
		int startRow = Row - P;
		//caculate the res
		for (int k = 0; k < K; k++) {
			double pixVal = 0;
			
			for (int i = 0; i < maskwidth; i++)
			{
				for (int j = 0; j < maskwidth; j++)
				{
					int curRow = startRow + i;
					int curCol = startCol + j;
					for (int c = 0; c < C; c++) {
						pixVal += mask[i*maskwidth*C*K + j*C*K + c*K+ k] * in[curRow*pad_H*C + curCol * C+ c];
						//printf("%.1f", mask[i*maskwidth*C*K + j * C*K + c * K + k]);
					}
				}
			}
			
			out[(Col-P)*w*K + (Row-P) * K + k] = pixVal;
			
		}
	}
}


//tiled shared memory gpu version
__global__
void convolution_2D_shared(double *in, double *out, double *mask, int maskwidth, int w, int h, int C, int K, int P, int pad_W, int pad_H) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_o = blockIdx.y*O_TILE_WIDTH + ty + P;
	int col_o = blockIdx.x*O_TILE_WIDTH + tx + P;
	int row_i = row_o - P;
	int col_i = col_o - P;
	__shared__ double Ns[BLOCK_WIDTH][BLOCK_WIDTH][IN_CHANNELS];
	for (int c = 0; c < C; c++) {
		Ns[ty][tx][c] = in[row_i*pad_H*C + col_i * C + c];
	}

	
	double output = 0.0f;
	
	for (int k = 0; k < K;k++) {
		for (int i = 0; i < maskwidth; i++) {
			for (int j = 0; j < maskwidth; j++) {
				for (int c = 0; c < C; c++) {
					output += mask[i*maskwidth*C*K + j * C*K + c * K + k] * Ns[i + row_i][j + row_i][c];
				}
			}
		}
	
		out[row_o*w*K + col_o*K+k] = output;
	}
}


__global__ void test()
{
	int Col = blockIdx.x*blockDim.x + threadIdx.x;
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	printf("%d,%d]\n", Row, Col);
	printf("%d,%d,%d)\n", blockDim.y, blockDim.x, blockDim.z);
	printf("%d,%d,%d)\n", gridDim.x, gridDim.y, gridDim.z);
}
int  main()
{
	int K = 64, FH = 3, FW = 3, H = 1024, W = 1024, C = 3, P = 1;
	int pad_H = H + 2 * P;
	int pad_W = W + 2 * P;
	printf("the mask(filter) size is :%d X %d.\n", filter_size, filter_size);
	printf("the matrix size is :%d X %d.\n", arr_size, arr_size);
	clock_t start_CPU, end_CPU;

	//filter
	double**** pFilter = new double***[FW];
	for (int i = 0; i<FW; i++) {
		pFilter[i] = new double**[FH];
		for (int j = 0; j < FH; j++) {
			pFilter[i][j] = new double*[C];
			for (int m = 0; m < C; m++) {
				pFilter[i][j][m] = new double[K];
				for (int n = 0; n < K; n++) {
					pFilter[i][j][m][n] = (double)(i + j)*(m + n);
				}
			}
		}
	}
	//image
	
	double*** arr = new double**[pad_W];
	for (int c = 0; c< pad_W; c++) {
		arr[c] = new double*[pad_H];
		for (int x = 0; x < pad_H; x++) {
			arr[c][x] = new double[C];
			
			for (int y = 0; y < C; y++) {
				if (c < P || c >= pad_W - P || x < P || x >= pad_H - P) {
					arr[c][x][y] = (double)0;
				}
				else {
					arr[c][x][y] = (double)y * (x + c);
				}
				
				
			}
		}
	}
	//output image
	double*** arr_out = new double**[W];
	for (int c = 0; c < W; c++) {
		arr_out[c] = new double*[H];
		for (int x = 0; x < H; x++) {
			arr_out[c][x] = new double[K];
			
		}
	}

	double res = Conv2(pFilter, arr, arr_out, FW, pad_H, C, K, P);
	printf("-------------------cpu version Done!------------------\n");
	printf("cpu res %.2f\n", res);
	//arr res pFilter size and memory allocation
	int arr_size_1D = C*pad_H*pad_W;
	int filter_size_1D = C*FH*FW*K;
	int arr_out_size_1D = K * H * W;
	double *arr_1D = (double*)malloc(arr_size_1D * sizeof(double));
	double *res_1D = (double*)malloc(arr_out_size_1D * sizeof(double));
	double *filter1D = (double*)malloc(filter_size_1D * sizeof(double));

	//IMAG_1D
	for (int i = 0; i<pad_W; i++) {
		for (int j = 0; j < pad_H; j++) {
			for (int c = 0; c < C; c++) {
				arr_1D[i*pad_H*C + j*C+c] = arr[i][j][c];
			}
		}
	}
	// filter_1D
	for (int k = 0; k < K; k++) {
		for (int i = 0; i<FW; i++) {
			for (int j = 0; j < FH; j++) {
				for (int c = 0; c < C; c++) {
				
						filter1D[i*FH*C*K + j * C*K + c * K + k] = pFilter[i][j][c][k];
				
				}
			}
		}
	}

	for (int i = 0; i< arr_size_1D; i++)
	{
		//printf("%.2lf ", arr_1D[i]);
	}
	//printf("\n");
	//GPU convolution_2D



	//allocate mem
	double *inD, *outD, *maskD;
	LARGE_INTEGER  num;
	long long start, end, freq, start2, end2, start3, end3;
	
	

	//malloc
	cudaMalloc((void**)&inD, sizeof(double)*arr_size_1D);
	cudaMalloc((void**)&outD, sizeof(double)*arr_out_size_1D);
	cudaMalloc((void**)&maskD, sizeof(double)*filter_size_1D);

	//copy
	cudaMemcpy(inD, arr_1D, sizeof(double)*arr_size_1D, cudaMemcpyHostToDevice);

	cudaMemcpy(maskD, filter1D, sizeof(double)*filter_size_1D, cudaMemcpyHostToDevice);

	//kerner function void convolution_2D_basic(float *in,float *out,float *mask,int maskwidth,int w,int h)
	
	int threadPerBlockX = 16;
	int threadPerBlockY = 16;
	dim3 grid((W - 1) / threadPerBlockX + 1,(H - 1) / threadPerBlockY + 1);
	dim3 block(threadPerBlockX, threadPerBlockY);
	QueryPerformanceCounter(&num);
	freq = num.QuadPart;
	start = num.QuadPart;
	convolution_2D_basic <<<grid, block >>>(inD, outD, maskD, FH, W, H, C, K, P, pad_W, pad_H);
	cudaMemcpy(res_1D, outD, sizeof(double)*arr_out_size_1D, cudaMemcpyDeviceToHost);
	printf("-------------------GPU Simple version Done!------------------\n");

	QueryPerformanceCounter(&num);
	end = num.QuadPart;
	printf("C1 time: %f ms\n", (end - start) * 1000 * 1.0 / freq);
	double check_res = 0.0;
	for (int i = 0; i < arr_out_size_1D; i++)
	{
		check_res += res_1D[i];
	}
	printf("C1 CHECKSUM res: %.2f\n", check_res);




	
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
	dim3 dimGrid((arr_size - 1) / O_TILE_WIDTH + 1, (arr_size - 1) / O_TILE_WIDTH + 1, 1);
	QueryPerformanceCounter(&num);
	start2 = num.QuadPart;
	convolution_2D_shared <<<dimGrid, dimBlock >> > (inD, outD, maskD, FH, W, H, C, K, P, pad_W, pad_H);
	cudaMemcpy(res_1D, outD, sizeof(double)*arr_out_size_1D, cudaMemcpyDeviceToHost);
	QueryPerformanceCounter(&num);
	end2 = num.QuadPart;
	printf("-------------------GPU Shared version Done!------------------\n");

	//check the res;
	//check(arr_1D,res_1D,arr_size_1D);

	check_res = 0.0;
	printf("C2 time: %f ms\n", (end2 - start2) * 1000 * 1.0 / freq);
	for (int i = 0; i < arr_out_size_1D; i++)
	{
		check_res += res_1D[i];
	}
	printf("C2 CHECKSUM: %.2f\n", check_res);
	cudaFree(outD);
	cudaFree(maskD);

	getchar();
}