#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#include "g.h"
#include "g_re.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;

bool Pretreatment(GpuMat* dSrcImage, GpuMat* dDstImage, int nSamples)
{
	GpuMat dGrayImage[nSamples],dStretchImage[nSamples],dBinaryImage[nSamples],dBlurImage[nSamples],dLaplacian[nSamples],
		   dEnhancedImage[nSamples];

	//Initialize
	for( int i = 0; i < nSamples; i++)
	{
		//creat
		dGrayImage[i].create(dSrcImage[i].size(),dSrcImage[i].type());
		dStretchImage[i].create(dSrcImage[i].size(),dSrcImage[i].type());
		dBinaryImage[i].create(dSrcImage[i].size(),dSrcImage[i].type());
		dBlurImage[i].create(dSrcImage[i].size(),dSrcImage[i].type());
		dLaplacian[i].create(dSrcImage[i].size(),dSrcImage[i].type());
		dEnhancedImage[i].create(dSrcImage[i].size(),dSrcImage[i].type());

		//set to 0
		dGrayImage[i].setTo(0,GpuMat());
		dStretchImage[i].setTo(0,GpuMat());
		dBinaryImage[i].setTo(0,GpuMat());
		dBlurImage[i].setTo(0,GpuMat());
		dLaplacian[i].setTo(0,GpuMat());
		dEnhancedImage[i].setTo(0,GpuMat());
	}

// //show to check
//	Mat tempImage[nSamples];
//	for(int i = 0; i < nSamples; i++){
//		dStretchImage[i].download(tempImage[i]);
//		cv::namedWindow("Original dStretchImage");
//		cv::imshow("Original dStretchImage",tempImage[i]);
//		cv::waitKey(1);
//	}

	//灰度化
	//launch_grayKernel(dSrcImage, dGray, classes, train_samples);
	//grayKernel<<<1,classes,train_samples>>>(dSrcImage, dGray);

	//灰度拉伸
	cout<<'\t'<<":Gray stretch"<<endl;
	launch_stretchKernel(dSrcImage, dStretchImage, nSamples);

// //show to check
//	Mat tempImage[nSamples];
//	for(int i = 0; i < nSamples; i++){
//		dStretchImage[i].download(tempImage[i]);
//		cv::namedWindow("dStretchImage");
//		cv::imshow("dStretchImage",tempImage[i]);
//		cv::waitKey(1);
//	}

	//二值处理
	cout<<'\t'<<":Binary processing"<<endl;
	launch_binaryKernel(dStretchImage, dBinaryImage, nSamples);

// //show to check
//	Mat tempImage[nSamples];
//	for(int i = 0; i < nSamples; i++){
//		dBinaryImage[i].download(tempImage[i]);
//		cv::namedWindow("dStretchImage");
//		cv::imshow("dStretchImage",tempImage[i]);
//		cv::waitKey(1);
//	}

	//锐化  createGaussianFilter_GPU
//	IplImage *pBlur8UC1 = cvCreateImage(cvGetSize(g_pBinaryImage),IPL_DEPTH_8U,1);
//	IplImage *pLablacian = cvCreateImage(cvGetSize(g_pBinaryImage),IPL_DEPTH_32F,1);
//	IplImage *pEnhanced = cvCreateImage(cvGetSize(g_pBinaryImage),IPL_DEPTH_8U,1);
	cout<<'\t'<<":Images enhancing "<<endl;
	for(int i = 0;i < nSamples; i++){

	//	cvSmooth(g_pBinaryImage,pBlur8UC1,CV_GAUSSIAN,3,3,0,0);
	//	CV_EXPORTS void GaussianBlur(const GpuMat& src, GpuMat& dst, Size ksize, double sigma1, double sigma2 = 0,
	//	                             int rowBorderType = BORDER_DEFAULT, int columnBorderType = -1);

		cv::gpu::GaussianBlur(dBinaryImage[i],dBlurImage[i],Size(3,3),0,0,BORDER_DEFAULT,-1);

//		cout<<"1"<<endl;
//		dBlurImage[i].download(tempImage[i]);
//		cv::namedWindow("dBlurImage");
//		cv::imshow("dBlurImage",tempImage[i]);
//		cv::waitKey(1000);


//		cvLaplace(pBlur8UC1,pLablacian,3);
//		CV_EXPORTS void Laplacian(const GpuMat& src, GpuMat& dst, int ddepth, int ksize = 1, double scale = 1, int borderType = BORDER_DEFAULT, Stream& stream = Stream::Null());
		//cv::gpu::Laplacian(dBlurImage[i],dBinaryImage[i],0,1,1,BORDER_DEFAULT,Stream::Null());
		cv::gpu::Laplacian(dBlurImage[i],dLaplacian[i],0,1,1,BORDER_DEFAULT,Stream::Null());

//		cout<<"2"<<endl;
//		dLaplacian[i].download(tempImage[i]);
//		cv::namedWindow("dLaplacian");
//		cv::imshow("dLaplacian",tempImage[i]);
//		cv::waitKey(1000);


//		cvConvert(pLablacian,pEnhanced);
//		void enqueueConvert(const GpuMat& src, GpuMat& dst, int dtype, double a = 1, double b = 0);
		cv::gpu::Stream stream;
		stream.enqueueConvert(dLaplacian[i],dEnhancedImage[i],dSrcImage->type());

//		cout<<"3"<<endl;
//		dEnhancedImage[i].download(tempImage[i]);
//		cv::namedWindow("dEnhancedImage");
//		cv::imshow("dEnhancedImage",tempImage[i]);
//		cv::waitKey(1000);


//	cvSub(pBlur8UC1,pEnhanced,pEnhanced,0);
//	CV_EXPORTS void subtract(const GpuMat& a, const GpuMat& b, GpuMat& c, const GpuMat& mask = GpuMat(), int dtype = -1, Stream& stream = Stream::Null());
		cv::gpu::subtract(dBlurImage[i],dEnhancedImage[i],dEnhancedImage[i],GpuMat(),-1,Stream::Null());

//		cout<<"4"<<endl;
//		dEnhancedImage[i].download(tempImage[i]);
//		cv::namedWindow("dEnhancedImage");
//		cv::imshow("dEnhancedImage",tempImage[i]);
//		cv::waitKey(1000);

	}

	//归一化处理
//	int NewHeight = 40;
//	int NewWidth = 20;
//	IplImage* image = cvCreateImage(cvSize(NewWidth,NewHeight),IPL_DEPTH_8U,1);
//	cvResize(pEnhanced,image);
//
//	return image;
	//gai


	return true;
}
