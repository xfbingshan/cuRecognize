#ifndef _G_H
#define _G_H

#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

//opencv
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

//extern "C" {
	void launch_binaryKernel(GpuMat * dStretch, GpuMat * dBinaryImage, int train_samples);
	void launch_stretchKernel(GpuMat * dGray,GpuMat * dStretch, int train_samples);
	void launch_characterKernel(GpuMat * dPreImage, float* character, int nSamples);
//}
#endif
