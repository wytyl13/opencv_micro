#ifndef _IMAGEENHANCEMENT_
#define _IMAGEENHANCEMENT_
#include "general.h"


double* getDistribution(const cv::Mat &inputImage);
void histogramEqualizeTransformation(const cv::Mat &inputImage, cv::Mat &outputImage);
#endif