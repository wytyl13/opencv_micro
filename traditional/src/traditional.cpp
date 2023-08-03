#include "traditional.h"

void thresholdAndLimitRange(cv::Mat &inputImage, cv::Rect rect)
{
    // we can get the subimage based on the rect. and any operation you have 
    // done in subimage will influence the original image.
    cv::Mat subImage = inputImage(rect);
    subImage.setTo(cv::Scalar(255, 255, 255));
}