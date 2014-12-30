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
	Mat tempImage[nSamples];
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

	int NewHeight = 40;
	int NewWidth = 20;
	//
	// To be optimized
	//
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

		//归一化处理
		cv::gpu::resize(dEnhancedImage[i], dDstImage[i], Size(NewWidth,NewHeight), 0, 0, INTER_LINEAR, Stream::Null());

//		dDstImage[i].download(tempImage[i]);
//		cv::namedWindow("dDstImage");
//		cv::imshow("dDstImage",tempImage[i]);
//		cv::waitKey(1);
	}
	return true;
}

//提取字符特征
bool Character(GpuMat* dPreImage,float ** character,int nSamples)
{


//	float sumMatValue(const Mat& image); // 计算图像中像素灰度值总和
//
//	float num_t[CHARACTER ]={0};  //定义存放字符特征值的数组num_t
//
//	int i=0,j=0,k=0;//循环变量
//	int Width = imgTest->width;//图像宽度
//	int Height = imgTest->height;//图像高度
//	int W = Width/4;//每小块的宽度
//	int H = Height/8;//每小块的宽度
//
//
//	//13点特征法:将图像平分为4*8的小块，统计每块中所有灰度值为255的点数
//
//	for(k=0; k<32; k++)
//	{
//		for(j=int(k/4)*H; j<int(k/4+1)*H; j++)
//		{
//			for(i=(k%4)*W;i<(k%4+1)*W;i++)
//			{
//			   num_t[k] += CV_IMAGE_ELEM(imgTest,uchar,j,i)/255 ;
//			}
//		}
//		num_t[32]+= num_t[k];  // 第33个特征：前32个特征的和作为第33个特征值,图像所有灰度值为255的点数
//
//		num_character[k] = num_t[k];
//	}
//	num_character[32] = num_t[32];
//
//
//	//垂直特征法：自左向右对图像进行逐列的扫描，统计每列白色像素的个数
//
//	for(i=0;i<Width;i++)
//	{
//		for(j=0;j<Height;j++)
//		{
//			num_t[33+i] += CV_IMAGE_ELEM(imgTest,uchar,j,i)/255 ;
//			num_character[33+i] = num_t[33+i];
//		}
//	}
//
//	//垂直特征法：自上而下逐行扫描，统计每行的黑色像素的个数
//
//	for(j=0;j<Height;j++)
//	{
//		for(i=0;i<Width;i++)
//		{
//			num_t[33+Width+j] += CV_IMAGE_ELEM(imgTest,uchar,j,i)/255;
//			num_character[33+Width+j] = num_t[33+Width+j];
//		}
//	}
//
//
//	//梯度分布特征
//
//	// 计算x方向和y方向上的滤波
//	float mask[3][3] = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };
//	Mat y_mask = Mat(3, 3, CV_32F, mask) / 8; //定义soble水平检测算子
//	Mat x_mask = y_mask.t(); // 转置,定义竖直方向梯度检测算子
//
//	//用x_mask和y_mask进行对字符图像进行图像滤波得到SobelX和SobelY
//	Mat sobelX, sobelY;
//	Mat image = imgTest;//IplImage * -> Mat,共享数据
//	filter2D(image, sobelX, CV_32F, x_mask);
//	filter2D(image, sobelY, CV_32F, y_mask);
//	sobelX = abs(sobelX);
//	sobelY = abs(sobelY);
//
//
//	//计算图像总的像素和
//	float totleValueX = sumMatValue(sobelX);
//	float totleValueY = sumMatValue(sobelY);
//
//	// 将图像划分为10*5共50个格子，计算每个格子里灰度值总和的百分比
//	int m=0;
//	int n=50;
//	for (int i = 0; i < image.rows; i = i + 4)
//	{
//		for (int j = 0; j < image.cols; j = j + 4)
//		{
//			Mat subImageX = sobelX(Rect(j, i, 4, 4));
//			num_t[33+Width+Height+m] = sumMatValue(subImageX) / totleValueX;
//			num_character[33+Width+Height+m] = num_t[33+Width+Height+m];
//			Mat subImageY= sobelY(Rect(j, i, 4, 4));
//			num_t[33+Width+Height+n] = sumMatValue(subImageY) / totleValueY;
//			num_character[33+Width+Height+n] = num_t[33+Width+Height+n];
//			m++;
//			n++;
//		}
//	}


	return true;
}
