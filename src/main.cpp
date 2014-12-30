#include <stdio.h>
#include <time.h>

#include <iostream>
#include <cstring>
#include <fstream>
#include <vector>

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

#define CHARACTER 193  //字符特征数

//float num_character[CHARACTER]={0};

//* -----------------------字符特征提取(垂特征直方向数据统计、13点法、梯度分布特征)-----------------------------------------//
// --Input：
//               IplImage *imgTest             // 字符图片
//
//-- Output:
//               int *num_t      // 字符特征
//--  Description:

//               1. 13点特征提取法:就是在字符图像中提取13个点作为特征向量。首先将图像平均分成8个方块,
//                  统计每个方块内的灰度为1的个数作为8个特征点,接下来在水平和垂直方向上分别画出三分之一和三分之二的位置,
//                  统计这些位置上的像素点作为4个特征点,最后统计图像中所有的灰度值为1的像素点个数作为1个特征,一共提取了
//                  13个特征。这种方法适应性较好,误差相对来说要小。
//
//               2. 垂直特征提取方法的算法：自左向右对图像进行逐列的扫描，统计每列白色像素的个数，然后自上而下逐行扫描，
//                  统计每行的黑色像素的个数，将统计的结果作为字符的特征向量，如果字符的宽度为 w,长度为 h,则特征向量
//                  的维数为 w+h.
//
//               3. 梯度分布特征：计算图像水平方向和竖直方向的梯度图像，然后通过给梯度图像分划不同的区域，
//                  进行梯度图像每个区域亮度值的统计，算法步骤为：
//                  <1>将字符由RGB转化为灰度，然后将图像归一化到40*20。
//                  <2>定义soble水平检测算子：x_mask=[−1,0,1;−2,0,2;–1,0,1]和竖直方向梯度检测算子y_mask=x_maskT。
//                  <3>对图像分别用mask_x和mask_y进行图像滤波得到SobelX和SobelY，下图分别代表原图像、SobelX和SobelY。
//                  <4>对滤波后的图像，计算图像总的像素和，然后划分4*2的网络，计算每个网格内的像素值的总和。
//                  <5>将每个网络内总灰度值占整个图像的百分比统计在一起写入一个向量，将两个方向各自得到的向量并在一起，组成特征向量。
//
//-------------------------------------------------------------------------*/


extern bool Pretreatment(GpuMat* dSrcImage, GpuMat* dDstImage, int nSamples);
extern bool Character(GpuMat* dPreImage,float ** character,int nSamples);

int main(int argc, char* argv[])
{
	//deviceQuery
	int deviceCount=cv::gpu::getCudaEnabledDeviceCount();
	if(0 != deviceCount){
		cout<<"CUDA capable device count: "<<deviceCount<<endl;
		cv::gpu::setDevice(0);
	}
	else{
		return -1;
	}

	//Timing
	static clock_t BeforeRunTime;//开始处理的时间
	clock_t UsedTime1,UsedTime2,UsedTime3,UsedTime4;//处理用去的时间

	BeforeRunTime = clock();
	cout<<(clock() - BeforeRunTime)*1000/CLOCKS_PER_SEC<<"ms"<<'\t'<<":Start"<<endl;

	//Parameters
	int train_samples = 128;//每类样本的个数
    int classes = 32;//样本种类
	int nSamples = train_samples*classes;//样本总数

	//file path
	//char file_path[50];//存放文件的路径
    char file[255];//文件名

	Mat classmat(1,nSamples,CV_32FC1);//目标分类结果的矩阵
	GpuMat dClassmat(1,nSamples,CV_32FC1);

	Mat srcImage[nSamples],dstImage[nSamples];
	GpuMat dSrcImage[nSamples];
  	Mat DataCharacter(nSamples, CHARACTER, CV_32FC1,Scalar(0));//创建大小为 样本总数*特征数 的矩阵DataCharacter，用来存放字符特征

  	for(int i = 0; i < classes; i++){
  	       // sprintf(file_path ,"%s/samples/%d/",getcwd(NULL, 0),
  	        for(int j = 1; j<= train_samples; j++)
  			{
  			    sprintf(file ,"%s/samples/%d/%d.png",getcwd(NULL, 0),i,j);
  			    srcImage[i*train_samples+j-1] = cv::imread(file,0);
  	            if(srcImage[i*train_samples+j-1].empty())
  				{
  					printf("Error: Can't load image %s\n", file);
  					return -2;
  				}
  	            dSrcImage[i*train_samples+j-1].upload(srcImage[i*train_samples+j-1]);
  	            //dSrcImage[i*train_samples+j-1].download(dstImage[i*train_samples+j-1]);

  	            classmat.at<float>(0,i*train_samples+j-1) = i;//记录目标分类结果

// 	            dSrcImage[i*train_samples+j-1].download(dstImage[i*train_samples+j-1]);
//  	            cv::namedWindow("x");
//  	            cv::imshow("x",dstImage[i*train_samples+j-1]);
//  	            cv::waitKey(1);
  	        }
  	    }
  	dClassmat.upload(classmat);
  	cout<<(clock() - BeforeRunTime)*1000/CLOCKS_PER_SEC<<"ms"<<'\t'<<":Images load succeed"<<endl;

  	//预处理
  	GpuMat dPreImage[nSamples];
  	cout<<(clock() - BeforeRunTime)*1000/CLOCKS_PER_SEC<<"ms"<<'\t'<<":Images pretreat start "<<endl;
  	Pretreatment(dSrcImage, dPreImage, nSamples);
  	cout<<(clock() - BeforeRunTime)*1000/CLOCKS_PER_SEC<<"ms"<<'\t'<<":Images pretreat succeed "<<endl;

  	//提取字符特征

  	float character[nSamples][CHARACTER];
  	cout<<(clock() - BeforeRunTime)*1000/CLOCKS_PER_SEC<<"ms"<<'\t'<<":Feature extraction start "<<endl;
	Character(dPreImage, (float **)character, nSamples);
	cout<<(clock() - BeforeRunTime)*1000/CLOCKS_PER_SEC<<"ms"<<'\t'<<":Feature extraction succeed "<<endl;

  	//yuanlaide
//    for(int i =0; i<classes; i++)
//    {
//
//			float* character = CodeCharacter(img);//提取字符特征
//			Mat tempMat = Mat(1, CHARACTER, CV_32FC1, character);//将特征数组转化成特征矩阵，以便于后续处理
//			Mat dsttemp = DataCharacter.row(i*train_samples+j-1);
//			tempMat.copyTo(dsttemp);//将每个样本的特征值作为一行存入特征矩阵中
//			cout<<DataCharacter.row(i*train_samples+j-1)<<endl;
//		}
//	}
//    UsedTime1 = (clock() - BeforeRunTime)*1000/CLOCKS_PER_SEC;
//
//	CvANN_MLP bp;//创建一个3层的神经网络，其中第一层结点数为x1,第二层结点数为x2，第三层结点数为x3
//	int x1 = CHARACTER;
//	int x2 = 85;
//	int x3 = classes;
//	int layer_num[3] = { x1, x2, x3 };
//
//	CvMat *layer_size = cvCreateMatHeader( 1, 3, CV_32S );
//	cvInitMatHeader( layer_size, 1, 3, CV_32S, layer_num );
//	bp.create( layer_size, CvANN_MLP::SIGMOID_SYM, 1, 1 );
//
//	//设定神经网络训练参数
//	CvANN_MLP_TrainParams params;
//	params.term_crit = cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 90000, 0.00001 );
//
//
//
//	params.train_method = CvANN_MLP_TrainParams::BACKPROP;
//	params.bp_dw_scale = 0.01;
//	params.bp_moment_scale = 0.05;
//
//   //	Mat outputs(1700,classes,CV_32FC1);//nSamples,classes,CV_32FC1);//目标输出矩阵
//	Mat outputs(nSamples,classes,CV_32FC1);//目标输出矩阵
//
//	//第i类所在的位置上的值最大为0.98，其他位置上的值较小，接近于0.02
//	for( int m = 0; m <  outputs.rows; m++ )
//	{
//		for( int k = 0; k < outputs.cols; k++ )
//        {
//			if( k == classmat.at<float>(0,m) )
//				outputs.at<float>(m,k) = 0.98;
//            else
//                outputs.at<float>(m,k) = 0.02;
//        }
//		cout<<outputs.row(m)<<endl;
//    }
//
//	//神经网络的训练
//
//	//Mat sampleWeights( 1, DataCharacter.rows, CV_32FC1, Scalar::all(1) );
//
//	Mat sampleWeights( 1, DataCharacter.rows, CV_32FC1);
//	randu(sampleWeights, Scalar(0), Scalar(1));
//
//	bp.train( DataCharacter, outputs, sampleWeights, Mat(), params );
//	//bp.train( DataCharacter, outputs, Mat(), Mat(), params );
//	printf(" 训练结束\n");
//	UsedTime2 = (clock() - BeforeRunTime)*1000/CLOCKS_PER_SEC;
//
//	//保存训练得到的权值矩阵
//    //bp.save("D:\\WorkSpace\\c++\\recognize1\\NN_DATA.xml");
//
//
//	/**************************************************
//
//	识别测试
//
//	**************************************************/
//
//
//	//字符识别
//	int test_samples = train_samples;
//	int test_classes = classes;
//	int ncounts = test_samples * test_classes;//1700;//nSamples;
//
//
//	//定义输出矩阵
//	Mat nearest(ncounts, test_classes, CV_32FC1, Scalar(0));
//	//Mat nearest(nSamples, classes, CV_32FC1, Scalar(0));
//
//	//神经网络识别
//	//bp.predict(ImgCharacter, nearest);
//	bp.predict(DataCharacter, nearest);
//
//
//	Mat result = Mat(1, ncounts, CV_32FC1);
//	for (int i=0;i<ncounts;i++)
//	{
//		Mat temp = Mat(1, CHARACTER, CV_32FC1);
//		Mat dsttempt = nearest.row(i);
//		dsttempt.copyTo(temp);
//		cout<<temp.row(0)<<endl;
//		Point maxLoc;
//		minMaxLoc(temp, NULL, NULL, NULL, &maxLoc);
//		result.at<float>(0,i) = maxLoc.x;
//	}
//	cout<<classmat.row(0)<<endl;
//	cout<<result.row(0)<<endl;
//	Mat resultd = Mat(1, ncounts, CV_32FC1);
//	resultd = classmat - result;
//
//	int countr = countNonZero(resultd);
//	float rate = 1-(float)countr/ncounts;
//	cout<<endl<<"Recognition rate: "<<rate<<endl; //统计识别率
//
//	UsedTime3 = (double)(clock() - BeforeRunTime)*1000/CLOCKS_PER_SEC;
//	printf("\nUsedTime1: %d ms\n",UsedTime1);
//	printf("\nUsedTime2: %d ms\n",UsedTime2-UsedTime1);
//	printf("\nUsedTime3: %d ms\n",UsedTime3-UsedTime2);

	return 0;
}


