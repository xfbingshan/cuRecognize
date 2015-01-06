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
#define CHARACTER 193

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
__global__ void stretchKernel3(PtrStepSzb* dGray, PtrStepSzb* dStretch, int* wMulh, double* dp, double* dp1, int* dnum, int nSamples, int threadsPerBlock)
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

	int threadsPerBlock = 512;
	int blockNum = (int) ceil( (float) nSamples / threadsPerBlock );
	init<<<blockNum, threadsPerBlock>>>(wMulh, dp, dp1, dnum, nSamples, threadsPerBlock);
	cudaDeviceSynchronize();

	// gray stretch, multiple blocks multiple threads

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

	stretchKernel3<<<blockNum, threadsPerBlock>>>(dG, dS, wMulh, dp ,dp1, dnum , nSamples, threadsPerBlock);

	cudaMemcpy(hG, dG, nSamples * sizeof(PtrStepSzb),cudaMemcpyDeviceToHost);
	cudaMemcpy(hS, dS, nSamples * sizeof(PtrStepSzb),cudaMemcpyDeviceToHost);

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
//			cv::namedWindow("StretchImage");
//			cv::imshow("StretchImage",tempImage[i]);
//			cv::waitKey(10);
//			cout<<i<<endl;
//	}

	// gray stretch, one block one thread
//	for(int i = 0 ; i < nSamples ; i++)
//	{
//		testKernel2<<<1, 1>>>(dGrayImage[i], dStretchImage[i], wMulh, dp ,dp1, dnum , nSamples, threadsPerBlock);
//	//	testKernel<<<1, (dGrayImage[i].cols,dGrayImage[i].rows)>>>(dGrayImage[i], dStretchImage[i], wMulh, dp ,dp1, dnum , nSamples, threadsP);
//	//	testKernel<<<1, block>>>(dGrayImage[0], dStretchImage[0], wMulh, dp ,dp1, dnum , nSamples, threadsP);
//	//	stretchKernel<<<blockNum, threadsP>>>(dGrayImage, dStretchImage, wMulh, dp ,dp1, dnum , nSamples, threadsP);
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
__device__ void Otsu(unsigned char * pImageData, int nWidth, int nHeight, int nWidthStep, unsigned int & dThreshold)
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
			dThreshold = j;
		}
	}
}

__device__ void myThreshold(PtrStepSzb src,PtrStepSzb & dst, unsigned int nThreshold)
{
	for(int y = 0; y < src.rows; y++)
		for(int x = 0 ; x < src.cols; x++)
		{
			if( src(y,x) > nThreshold)
				dst(y,x) = 255;
			else
				dst(y,x) = 0;
		}
}

__global__ void binaryKernel2(PtrStepSzb* src, PtrStepSzb* dst, unsigned int * dThreshold , int nSamples)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < nSamples){
		Otsu(src[tid].data, src[tid].cols, src[tid].rows, src[tid].step, dThreshold[tid]);
		myThreshold(src[tid], dst[tid], dThreshold[tid]);
	}
}

//二值处理
//extern "C"
void launch_binaryKernel(GpuMat * dStretchImage, GpuMat * dBinaryImage, int nSamples)
{
	// GpuMat to PtrStepSzb
	PtrStepSzb * dS = NULL;
	PtrStepSzb * dB = NULL;

	PtrStepSzb hS[nSamples];
	PtrStepSzb hB[nSamples];

	for( int i = 0 ; i < nSamples ; i++)
	{
		hS[i].cols = dStretchImage[i].cols;
		hS[i].rows = dStretchImage[i].rows;
		hS[i].data = dStretchImage[i].data;
		hS[i].step = dStretchImage[i].step;

		hB[i].cols = dBinaryImage[i].cols;
		hB[i].rows = dBinaryImage[i].rows;
		hB[i].data = dBinaryImage[i].data;
		hB[i].step = dBinaryImage[i].step;
	}

	cudaMalloc((void **)&dS, nSamples * sizeof(PtrStepSzb));
	cudaMalloc((void **)&dB, nSamples * sizeof(PtrStepSzb));

	cudaMemcpy(dS, hS, nSamples * sizeof(PtrStepSzb),cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, nSamples * sizeof(PtrStepSzb),cudaMemcpyHostToDevice);

	// threshold
	unsigned int * hThreshold = new unsigned int[nSamples];
	unsigned int * dThreshold = NULL;

	cudaMalloc((void **)&dThreshold, nSamples * sizeof(unsigned int));

	// kernel
	int threadsPerBlock = 512;
	int blockNum = (int) ceil( (float) nSamples / threadsPerBlock );
	binaryKernel2<<<blockNum, threadsPerBlock>>>(dS, dB, dThreshold, nSamples);

	cudaMemcpy(hS, dS, nSamples * sizeof(PtrStepSzb),cudaMemcpyDeviceToHost);
	cudaMemcpy(hB, dB, nSamples * sizeof(PtrStepSzb),cudaMemcpyDeviceToHost);

	// PtrStepSzb to GpuMat
	for( int i = 0 ; i < nSamples ; i++)
	{
		dStretchImage[i].cols = hS[i].cols;
		dStretchImage[i].rows = hS[i].rows;
		dStretchImage[i].data = hS[i].data;
		dStretchImage[i].step = hS[i].step;

		dBinaryImage[i].cols = hB[i].cols;
		dBinaryImage[i].rows = hB[i].rows;
		dBinaryImage[i].data = hB[i].data;
		dBinaryImage[i].step = hB[i].step;

	}

//	//show to check
//	Mat tempImage[nSamples];
//	for(int i = 0; i < nSamples; i++){
//			dBinaryImage[i].download(tempImage[i]);
//			cv::namedWindow("binaryImage");
//			cv::imshow("binaryImage",tempImage[i]);
//			cv::waitKey(10);
//			cout<<i<<endl;
//	}

}


__device__ void sumValue(PtrStepSzb dImage, float & sum)
{
	sum = 0;
	int r = dImage.rows;
	int c = dImage.cols;

	for (int i = 0; i < r; i++)
	{
		for (int j = 0; j < c; j++)
		{
			sum += dImage(i,j);
		}
	}

}

__global__ void characterKernel(PtrStepSzb * dPreImage, float * character, int nSamples)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < nSamples){
		float sum;
		sumValue(dPreImage[tid],sum);	// 计算图像中像素灰度值总和

		float num_t[CHARACTER] = {0}; //定义存放字符特征值的数组num_t

		int i=0,j=0,k=0;//循环变量
		int Width = dPreImage[tid].cols;//图像宽度
		int Height = dPreImage[tid].rows;//图像高度
		int W = Width/4;//每小块的宽度
		int H = Height/8;//每小块的高度

		//13点特征法:将图像平分为4*8的小块，统计每块中所有灰度值为255(白色像素)的点数

		for(k = 0; k < 32; k++)
		{
			for(j = int(k / 4) * H; j<int(k / 4 + 1) * H; j++)
			{
				for(i = (k % 4) * W;i < (k % 4 + 1) * W; i++)
				{
					num_t[k] += dPreImage[tid](j,i) / 255 ;
				}
			}
			num_t[32] += num_t[k];  // 第33个特征：前32个特征的和作为第33个特征值,图像所有灰度值为255(白色像素)的点数

			character[tid * CHARACTER + k] = num_t[k];
		}
		character[tid * CHARACTER + 32] = num_t[32];

		//垂直特征法：自左向右对图像进行逐列的扫描，统计每列白色像素的个数

		for(i = 0; i < Width; i++)
		{
			for(j = 0; j < Height; j++)
			{
				num_t[33 + i] += dPreImage[tid](j,i)/255 ;
				character[tid * CHARACTER + 33 + i] = num_t[33 + i];
			}
		}

		//垂直特征法：自上而下逐行扫描，统计每行的黑色像素(????)的个数

		for(j = 0; j < Height; j++)
		{
			for(i = 0; i < Width; i++)
			{
				num_t[33 + Width + j] += dPreImage[tid](j,i) / 255;
				character[tid * CHARACTER + 33 + Width + j] = num_t[33 + Width + j];
			}
		}

		//梯度分布特征

////	计算x方向和y方向上的滤波
//		float mask[3][3] = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };
//		Mat y_mask = Mat(3, 3, CV_32F, mask) / 8; //定义soble水平检测算子
//		Mat x_mask = y_mask.t(); // 转置,定义竖直方向梯度检测算子
//
//		//用x_mask和y_mask进行对字符图像进行图像滤波得到SobelX和SobelY
//		Mat sobelX, sobelY;
//		Mat image = imgTest;//IplImage * -> Mat,共享数据
//		filter2D(image, sobelX, CV_32F, x_mask);
//		filter2D(image, sobelY, CV_32F, y_mask);
//		sobelX = abs(sobelX);
//		sobelY = abs(sobelY);
//
//
//		//计算图像总的像素和
//		float totleValueX = sumMatValue(sobelX);
//		float totleValueY = sumMatValue(sobelY);
//
//		// 将图像划分为10*5共50个格子，计算每个格子里灰度值总和的百分比
//		int m=0;
//		int n=50;
//		for (int i = 0; i < image.rows; i = i + 4)
//		{
//			for (int j = 0; j < image.cols; j = j + 4)
//			{
//				Mat subImageX = sobelX(Rect(j, i, 4, 4));
//				num_t[33+Width+Height+m] = sumMatValue(subImageX) / totleValueX;
//				num_character[33+Width+Height+m] = num_t[33+Width+Height+m];
//				Mat subImageY= sobelY(Rect(j, i, 4, 4));
//				num_t[33+Width+Height+n] = sumMatValue(subImageY) / totleValueY;
//				num_character[33+Width+Height+n] = num_t[33+Width+Height+n];
//				m++;
//				n++;
//			}
//		}


	}



}


__global__ void testKernel(float* character)
{
	int tid =  threadIdx.x;
//	float a = 250.0f;
//	if(tid < nSamples){
//		for(int i = 0 ; i< CHARACTER; i++)
//		{
	float temp = 250;
	character[tid] = tid;
	printf("%d\t:%f\t%f\n",tid,temp,character[tid]);
//	character[tid] = tid;
//		}
//	}
}

void launch_characterKernel(GpuMat * dPreImage, float* character, int nSamples)
{
	// GpuMat to PtrStepSzb
	PtrStepSzb * dP = NULL;
	PtrStepSzb hP[nSamples];

	for( int i = 0 ; i < nSamples ; i++)
	{
		hP[i].cols = dPreImage[i].cols;
		hP[i].rows = dPreImage[i].rows;
		hP[i].data = dPreImage[i].data;
		hP[i].step = dPreImage[i].step;
	}

	cudaMalloc((void **)&dP, nSamples * sizeof(PtrStepSzb));
	cudaMemcpy(dP, hP, nSamples * sizeof(PtrStepSzb),cudaMemcpyHostToDevice);

//		//show to check
////		Mat tempImage[nSamples];
////		for(int i = 0; i < nSamples; i++){
////				dPreImage[i].download(tempImage[i]);
////				cv::namedWindow("PreImage");
////				cv::imshow("PreImage",tempImage[i]);
////				cv::waitKey(10);
////				cout<<i<<endl;
////		}
//
//		Mat tempImage[nSamples];
//		for(int i = 0; i < 1; i++){
//				dPreImage[i].download(tempImage[i]);
//				for(int j = 0; j < tempImage[i].rows; j++){
//					for (int k = 0; k < tempImage[i].cols; k++)
//						printf("%d\t",tempImage[i].at<uchar>(j,k));
//
//					cout<<endl;
//				}
//		}

	//
	int threadsPerBlock = 512;
	int blockNum = (int) ceil( (float) nSamples / threadsPerBlock );

//
//	float * c;
//	float hc[CHARACTER]={0};
//	cudaMalloc((void **)&c,CHARACTER * sizeof(float));
//	cudaMemcpy(c, hc, CHARACTER * sizeof(float),cudaMemcpyHostToDevice);
//	testKernel<<<1,CHARACTER>>>(c);
//	cudaMemcpy(hc, c, CHARACTER * sizeof(float),cudaMemcpyDeviceToHost);


	characterKernel<<<blockNum, threadsPerBlock>>>(dP, character, nSamples);


//	float testCharacter[nSamples * CHARACTER];
//	cudaMemcpy((void *)testCharacter, (void *)character, nSamples * CHARACTER * sizeof(float),cudaMemcpyDeviceToHost);
//	for(int i = 0; i < 1 ; i++)
//	{
//		for(int j = 0; j < CHARACTER ; j++)
//			printf("(%d,%d):\t%f\n",i,j,testCharacter[i * CHARACTER + j]);
//			//cout<<(float)testCharacter[i * CHARACTER + j];
//		cout<<endl;
//	}


//	cudaMemcpy(hS, dS, nSamples * sizeof(PtrStepSzb),cudaMemcpyDeviceToHost);
//	cudaMemcpy(hB, dB, nSamples * sizeof(PtrStepSzb),cudaMemcpyDeviceToHost);

	// PtrStepSzb to GpuMat
//	for( int i = 0 ; i < nSamples ; i++)
//	{
//		dStretchImage[i].cols = hS[i].cols;
//		dStretchImage[i].rows = hS[i].rows;
//		dStretchImage[i].data = hS[i].data;
//		dStretchImage[i].step = hS[i].step;
//
//		dBinaryImage[i].cols = hB[i].cols;
//		dBinaryImage[i].rows = hB[i].rows;
//		dBinaryImage[i].data = hB[i].data;
//		dBinaryImage[i].step = hB[i].step;
//
//	}

//	//show to check
//	Mat tempImage[nSamples];
//	for(int i = 0; i < nSamples; i++){
//			dBinaryImage[i].download(tempImage[i]);
//			cv::namedWindow("binaryImage");
//			cv::imshow("binaryImage",tempImage[i]);
//			cv::waitKey(10);
//			cout<<i<<endl;
//	}


}

