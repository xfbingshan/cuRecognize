// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

////opencv256
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include "opencv2/core/core.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"


//
using namespace std;
using namespace cv;
using namespace cv::gpu;

//#define blocks 7
//#define threadsPerB 256
//#define size blocks * threadsPerB


__global__ void init(int* wMulh, double* dp, double* dp1, int* dnum, int nSamples, int threadsPerBlock){
	int tid =  blockIdx.x * threadsPerBlock + threadIdx.x;
	if(tid < nSamples)
	{
		wMulh[tid] = 0;
		for( int i = 0; i < 256; i++){
			dp[tid * 256 + i]   = 0;
			dp1[tid * 256 + i]  = 0;
			dnum[tid * 256 + i] = 0;
		}

//		printf("Init: %d\t%g\n",tid, dp[tid * 256 + 255]);
	}
}

__global__ void stretchKernel(GpuMat* dGray, GpuMat* dStretch, int* wMulh, double* dp, double* dp1, int* dnum, int nSamples, int threadsPerBlock)
{

	int tid = blockIdx.x * threadsPerBlock + threadIdx.x;

	if(tid < nSamples){
		wMulh[tid] = dGray[tid].cols * dGray[tid].rows;

//		printf("Y1:%d\t%d\n",tid, wMulh[tid]);

		for(int x = 0;x < dGray[tid].rows; x++)
		{
//			printf("Y2:%d\t%d", tid, dGray[tid].step * x );
			for(int y = 0;y < dGray[tid].cols; y++){
				uchar v = ((uchar*)(dGray[tid].data + dGray[tid].step * x))[y];
				dnum[tid * 256 + v]++;
//				printf("Y3:%d\t%d", tid, dGray[tid].step * x + y );
			}
		}


//		for(int i = 0; i < 256 ; i++)
//		{
//			if(dnum[tid * 256 + i] != 0)
//				printf("Y:%d\t%g\n",tid, dnum[tid * 256]);
//		}

		//calculate probability
		for(int i = 0;i < 256; i++)
		{
			dp[tid * 256 + i] = (double)dnum[tid * 256 + i] / (double)wMulh[tid];
		}
		 //p1[i]=sum(p[j]);  j<=i;
		for(int i = 0;i < 256; i++)
		{
			for(int k = 0;k <= i; k++)
				dp1[tid * 256 + i] += dp[tid * 256 + k];
		}

		// histogram transformation
		for(int x = 0;x < dGray[tid].rows; x++)
		{
			for(int y = 0;y < dGray[tid].cols; y++){
				uchar v = ((uchar*)(dGray[tid].data + dGray[tid].step * x))[y];
				((uchar*)(dStretch[tid].data + dStretch[tid].step * x))[y] = dp1[tid * 256 + v]*255 + 0.5;
			}
		}

	}
//	printf("%d\n",tid);
}

__global__ void swap_rb_kernel(const PtrStepSz<uchar3> src,PtrStep<uchar3> dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < src.cols && y < src.rows)
    {
        uchar3 v = src(y,x);
        dst(y,x) = make_uchar3(v.z,v.y,v.x);
    }
}

void swap_rb_caller(const PtrStepSz<uchar3>& src,PtrStep<uchar3> dst,cudaStream_t stream)
{
    dim3 block(32,8);
    dim3 grid((src.cols + block.x - 1)/block.x,(src.rows + block.y - 1)/block.y);

    swap_rb_kernel<<<grid,block,0,stream>>>(src,dst);
    if(stream == 0)
        cudaDeviceSynchronize();
}


__global__ void testKernel(PtrStepSzb dGray, PtrStepSzb dStretch, int* wMulh, double* dp, double* dp1, int* dnum, int nSamples, int threadsPerBlock)
{

	int x = //blockIdx.x * blockDim.x +
			threadIdx.x;
	int y = //blockIdx.y * blockDim.y +
			threadIdx.y;

	const int tid = blockDim.x * y + x;

	printf("1\n");
	if(x < dGray.cols && y < dGray.rows){
//		dStretch(y,x) = 255;//dGray(x,y);
		wMulh[0] = dGray.cols * dGray.rows;

		uchar v = dGray.ptr(y)[x];
		dnum[v]++;
	}

	//calculate probability
	__syncthreads();
	printf("2\n");
	if(x < 32 && y < 8){
		dp[tid] = (double)dnum[tid] / (double)wMulh[0];
	}

	//p1[i]=sum(p[j]);  j<=i;
	__syncthreads();
	printf("3\n");
	if(x < 32 && y < 8){
		for(int k = 0;k <= tid; k++)
			dp1[tid] += dp[k];
	}
	// histogram transformation
	__syncthreads();
	printf("4\n");
	if(x < 32 && y < 8){
		uchar v = dGray.ptr(y)[x];
		dStretch.ptr(y)[x] = dp1[v]*255 + 0.5;
	}
}

//single picture , one block one thread
__global__ void testKernel2(PtrStepSzb dGray, PtrStepSzb dStretch, int* wMulh, double* dp, double* dp1, int* dnum, int nSamples, int threadsPerBlock)
{
	wMulh[0] = dGray.cols * dGray.rows;

	for(int y = 0;y < dGray.rows; y++){
		for(int x = 0;x < dGray.cols ; x++){
			uchar v = dGray(y,x);
			dnum[v]++;
		}
	}

	//calculate probability
	for(int i = 0;i < 256; i++)
	{
		dp[i] = (double)dnum[i] / (double)wMulh[0];
	}

	//p1[i]=sum(p[j]);  j<=i;
	for(int i = 0;i < 256; i++){
		for(int k = 0; k <= i ; k++){
			dp1[i] += dp[k];
		}
	}

	// histogram transformation
	for(int y = 0;y < dGray.rows; y++){
		for(int x = 0;x < dGray.cols ; x++){
			uchar v = dGray(y,x);
			dStretch(y,x) = dp1[v]*255 + 0.5;
		}
	}

	//clean

	for(int i = 0 ; i < 256 ; i++)
	{
		dnum[i] = 0;
		dp[i] = 0;
		dp1[i] = 0;
	}
}

// one block multiple threads
__global__ void testKernel3(PtrStepSzb* dGray, PtrStepSzb* dStretch, int* wMulh, double* dp, double* dp1, int* dnum, int nSamples, int threadsPerBlock)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int cols = dGray[tid].cols;
	int rows = dGray[tid].rows;

	wMulh[tid] = cols * rows;

	for(int y = 0;y < rows; y++){
		for(int x = 0;x < cols ; x++){
			uchar v = dGray[tid](y,x);
			dnum[tid * 256 + v]++;
		}
	}

	//calculate probability
	for(int i = 0;i < 256; i++)
	{
		dp[tid * 256 + i] = (double)dnum[tid * 256 + i] / (double)wMulh[tid];
	}

	//p1[i]=sum(p[j]);  j<=i;
	for(int i = 0;i < 256; i++){
		for(int k = 0; k <= i ; k++){
			dp1[tid * 256 + i] += dp[tid * 256 + k];
		}
	}

	// histogram transformation
	for(int y = 0;y < rows; y++){
		for(int x = 0;x < cols ; x++){
			uchar v = dGray[tid](y,x);
			dStretch[tid](y,x) = dp1[tid * 256 + v]*255 + 0.5;
		}
	}

	//clean
	for(int i = 0 ; i < 256 ; i++)
	{
		dnum[tid * 256 + i] = 0;
		dp[tid * 256 + i] = 0;
		dp1[tid * 256 + i] = 0;
	}
}



//灰度拉伸
//extern "C"
void launch_stretchKernel(GpuMat * dGrayImage,GpuMat * dStretchImage, int nSamples)
{
	int * wMulh  = NULL;
	double *dp   = NULL;
	double *dp1  = NULL;
	int *dnum = NULL;
	cudaMalloc( (void**)&wMulh, nSamples * 256 * sizeof(int)    );
	cudaMalloc( (void**)&dp   , nSamples * 256 * sizeof(double) );
	cudaMalloc( (void**)&dp1  , nSamples * 256 * sizeof(double) );
	cudaMalloc( (void**)&dnum , nSamples * 256 * sizeof(int)    );

	int * num  = (int *)calloc(nSamples * 256 , sizeof(int));
	double * p = (double *)calloc(nSamples * 256 , sizeof(double));
//	Mat tempImage[nSampexpectedles];
//	for(int i = 0; i < nSamples; i++){
//		dGray[i].download(tempImage[i]);
//		cv::namedWindow("x");
//		cv::imshow("x",tempImage[i]);
//		cv::waitKey(1);
//	}

	int threadsPerBlock = 512;
	int blockNum = (int) ceil( (float) nSamples / threadsPerBlock );
	init<<<blockNum, threadsPerBlock>>>(wMulh, dp, dp1, dnum, nSamples, threadsPerBlock);
	cudaDeviceSynchronize();

//	Mat tempImage[nSamples];
//	for(int i = 0; i < nSamples; i++){
//		dStretchImage[i].download(tempImage[i]);
//		cv::namedWindow("x");
//		cv::imshow("x",tempImage[i]);
//		cv::waitKey(1);
//	}
//	printf("type:%d\n",dGrayImage->type());


//	uchar * dG1 = dGrayImage->operator cv::gpu::PtrStepSz<uchar>();

//
//	for(int i = 0 ; i < nSamples ; i++){
//			hG[i]= dGrayImage[i];//.operator cv::gpu::PtrStepSz<uchar>();//(PtrStepSzb)dGrayImage[i];
//			hS[i]= dStretchImage[i];//.operator cv::gpu::PtrStepSz<uchar>();//(PtrStepSzb)dStretchImage[i];
//			}
//

//

//
//	cudaMemcpy(&dG,&hG,nSamples,cudaMemcpyHostToDevice);
//	cudaMemcpy(&dS,&hS,nSamples,cudaMemcpyHostToDevice);
//
////	for(int i = 0 ; i < nSamples ; i++){
////		dG[i]= dGrayImage[i];//.operator cv::gpu::PtrStepSz<uchar>();//(PtrStepSzb)dGrayImage[i];
////		dS[i]= dStretchImage[i];//.operator cv::gpu::PtrStepSz<uchar>();//(PtrStepSzb)dStretchImage[i];
////		}
//
//	dim3 block(32,32);

	// gray stretch, one block multiple threads

//	cout<<"dG data size:"<<sizeof(dG->data)<<endl;
//	cout<<"size_t size:"<<sizeof(size_t)<<endl;
//	cout<<"PtrStepSzb size:"<<sizeof(PtrStepSzb)<<endl;
//	cout<<"GpuMat size:"<<sizeof(GpuMat)<<endl;

	PtrStepSzb * dG = NULL;
	PtrStepSzb * dS = NULL;
	PtrStepSzb hG[nSamples];
	PtrStepSzb hS[nSamples];
	for( int i = 0 ; i < nSamples ; i++)
	{
		hG[i].cols = dGrayImage[i].cols;
		hG[i].rows = dGrayImage[i].rows;
		hG[i].data = dGrayImage[i].data;
		hG[i].step = dGrayImage[i].step;

		hS[i].cols = dStretchImage[i].cols;
		hS[i].rows = dStretchImage[i].rows;
		hS[i].data = dStretchImage[i].data;
		hS[i].step = dStretchImage[i].step;
	}
	cudaMalloc((void **)&dG, nSamples * sizeof(PtrStepSzb));
	cudaMalloc((void **)&dS, nSamples * sizeof(PtrStepSzb));
	cudaMemcpy(dG, hG, nSamples * sizeof(PtrStepSzb),cudaMemcpyHostToDevice);
	cudaMemcpy(dS, hS, nSamples * sizeof(PtrStepSzb),cudaMemcpyHostToDevice);

	testKernel3<<<blockNum, threadsPerBlock>>>(dG, dS, wMulh, dp ,dp1, dnum , nSamples, threadsPerBlock);

	for( int i = 0 ; i < nSamples ; i++)
	{
		dGrayImage[i].cols = hG[i].cols;
		dGrayImage[i].rows = hG[i].rows;
		dGrayImage[i].data = hG[i].data;
		dGrayImage[i].step = hG[i].step;

		dStretchImage[i].cols = hS[i].cols;
		dStretchImage[i].rows = hS[i].rows;
		dStretchImage[i].data = hS[i].data;
		dStretchImage[i].step = hS[i].step;
	}

//	//show to check
//	Mat tempImage[nSamples];
//	for(int i = 0; i < nSamples; i++){
//	//		tempImage[0].create(32,16,CV_8UC1);
//	//		dStretchImage[0].upload(tempImage[0]);
//			//cout<<"YYY:"<<dStretchImage[i].isContinuous()<<endl;//'\t'<<dStretchImage[0].<<'\t'<<dStretchImage[0].step<<endl;
//			dStretchImage[i].download(tempImage[i]);
//			cv::namedWindow("x");
//			cv::imshow("x",tempImage[i]);
//			cv::waitKey(50);
//			cout<<i<<endl;
//	}


	// gray stretch, one block one thread
	for(int i = 0 ; i < nSamples ; i++)
	{
		testKernel2<<<1, 1>>>(dGrayImage[i], dStretchImage[i], wMulh, dp ,dp1, dnum , nSamples, threadsPerBlock);
	//	testKernel<<<1, (dGrayImage[i].cols,dGrayImage[i].rows)>>>(dGrayImage[i], dStretchImage[i], wMulh, dp ,dp1, dnum , nSamples, threadsP);
	//	testKernel<<<1, block>>>(dGrayImage[0], dStretchImage[0], wMulh, dp ,dp1, dnum , nSamples, threadsP);
	//	stretchKernel<<<blockNum, threadsP>>>(dGrayImage, dStretchImage, wMulh, dp ,dp1, dnum , nSamples, threadsP);
	}


	unsigned int * nThreshold = new unsigned int[nSamples];

//	cudaDeviceSynchronize();
//	cudaMemcpy(num,dnum,256*sizeof(int),cudaMemcpyDeviceToHost);
//	cudaMemcpy(p,dp1,256*sizeof(double),cudaMemcpyDeviceToHost);

//	cout<<"num:\t";
//	for(int i = 0;i < 256 ;i++)
//		cout<<num[i]<<" ";
//	cout<<endl;
//
//	cout<<"p:\t";
//	for(int i = 0;i < 256 ;i++)
//		cout<<p[i]<<" ";
//	cout<<endl;


//
	//	dStretchImage[i].data = dS[i].data;}

//	cudaMemcpy(&hG,&dG,nSamples,cudaMemcpyDeviceToHost);
//	cudaMemcpy(&hS,&dS,nSamples,cudaMemcpyDeviceToHost);

//	Mat tempImage[nSamples];


//	for(int i = 0 ; i < nSamples ; i++){
//		for( int j = 0 ; j < dS[i].rows; j++)
//			for( int k = 0 ; k < dS[i].cols; k++)
//				tempImage[i].at<uchar>(j,k) = hS[i](j,k);
//	}

//	for(int i = 0 ; i < nSamples ; i++){
//				dG[i]= dGrayImage[i];//.operator cv::gpu::PtrStepSz<uchar>();//(PtrStepSzb)dGrayImage[i];
//				dS[i]= dStretchImage[i];//.operator cv::gpu::PtrStepSz<uchar>();//(PtrStepSzb)dStretchImage[i];
//				}

//	for(int i = 0; i < nSamples; i++){
////		tempImage[0].create(32,16,CV_8UC1);
////		dStretchImage[0].upload(tempImage[0]);
//		//cout<<"YYY:"<<dStretchImage[i].isContinuous()<<endl;//'\t'<<dStretchImage[0].<<'\t'<<dStretchImage[0].step<<endl;
//		dStretchImage[i].download(tempImage[i]);
//		cv::namedWindow("x");
//		cv::imshow("x",tempImage[i]);
//		cv::waitKey(10);
//		cout<<i<<endl;
//	}
}

__device__ void GetHistogram(unsigned char * pImageData, int nWidth, int nHeight, int nWidthStep, int * pHistogram)
{
	int i = 0;
	int j = 0;
	unsigned char *pLine = NULL;

	// 清空直方图
//	std::memset(pHistogram,0,sizeof(int) * 256);
	for (pLine = pImageData, j = 0; j < nHeight; j++, pLine += nWidthStep)
	{
		for (i = 0; i < nWidth; i++)
		{
			pHistogram[pLine[i]]++;
		}
	}

}
__device__ void Otsu(unsigned char * pImageData, unsigned int * nThreshold, int nWidth, int nHeight, int nWidthStep)
{
	int    i          = 0;
	int    j          = 0;
	int    nTotal     = 0;
	int    nSum       = 0;
	int    A          = 0;
	int    B          = 0;
	double u          = 0;
	double v          = 0;
	double dVariance  = 0;
	double dMaximum   = 0;
//	int    nThreshold = 0;
	int    nHistogram[256]={0};
	// 获取直方图
	GetHistogram(pImageData, nWidth, nHeight, nWidthStep, nHistogram);
	for (i = 0; i < 256; i++)
	{
		nTotal += nHistogram[i];
		nSum   += (nHistogram[i] * i);
	}
	for (j = 0; j < 256; j++)
	{
		A = 0;
		B = 0;
		for (i = 0; i < j; i++)
		{
			A += nHistogram[i];
			B += (nHistogram[i] * i);
		}
		if (A > 0)
		{
			u = B / A;
		}
		else
		{
			u = 0;
		}
		if (nTotal - A > 0)
		{
			v = (nSum - B) / (nTotal - A);
		}
		else
		{
			v = 0;
		}
		dVariance = A * (nTotal - A) * (u - v) * (u - v);
		if (dVariance > dMaximum)
		{
			dMaximum = dVariance;
			*nThreshold = j;
		}
	}
}

__device__ void myThreshold(GpuMat * src,GpuMat * dst, unsigned int nThreshold)
{
	for(int i = 0 ; i < src->cols; i++)
		for(int j = 0; j < src->rows; j++)
		{
			if( *(src->datastart + i * src->step + j) > nThreshold)
				*(dst->datastart + i * dst->step + j) = 255;
			else
				*(dst->datastart + i * dst->step + j) = 0;
		}
}

__global__ void binaryKernel(GpuMat * src, GpuMat * dst, unsigned int * nThreshold , int nSamples, int threadsPerBlock)
{
	const int tid = blockIdx.x * threadsPerBlock + threadIdx.x;

	if(tid < nSamples){
		Otsu(src[tid].data, &nThreshold[tid], src[tid].cols, src[tid].rows, src[tid].step);
		myThreshold(&src[tid], &dst[tid], nThreshold[tid]);
		printf("%u\n",nThreshold[tid]);
	}
}
/*
__global__ void binaryKernel2(PtrStepSzb src, PtrStepSzb dst, unsigned int & nThreshold , int nSamples, int threadsPerBlock)
{
	const int tid = blockIdx.x * threadsPerBlock + threadIdx.x;

	if(tid < nSamples){
		Otsu(src[tid].data, &nThreshold[tid], src[tid].cols, src[tid].rows, src[tid].step);
		myThreshold(&src[tid], &dst[tid], nThreshold[tid]);
		printf("%u\n",nThreshold[tid]);
	}
}*/

//二值处理
//extern "C"
void launch_binaryKernel(GpuMat * dStretch, GpuMat * dBinaryImage, int nSamples)
{
	unsigned int * nThreshold = new unsigned int[nSamples];

//	int threadsPerBlock = 256;
//	int blockNum = (int) ceil( (float) nSamples / threadsPerBlock );
//	binaryKernel<<<blockNum, threadsPerBlock>>>(dStretch, dBinaryImage, nThreshold, nSamples, threadsPerBlock);
//	for(int i = 0 ; i < nSamples; i++)
//	{
//		binaryKernel2<<<1, 1>>>(dStretch, dBinaryImage, nThreshold, nSamples, threadsPerBlock);
//	}
}
