/**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-04-17 13:39:21
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-04-17 13:39:21
 * @Description: the steps can be subdivided as follows.
 * 1. face aligment
 *      face landmark detection.
 *      calculate the convex hull.
 *      delaunary.
 *      affine transformation.
 * 2. seamless integration.
***********************************************************************/
#include "dlib.h"


int main(int argc, char const *argv[])
{
    if (argc != 3)
	{
		sys_error("please input the image and the parameters, the first param is srcImage, the second param is dest image.\n");
	}
    cv::Mat srcImage = cv::imread(argv[1]);
    cv::Mat destImage = cv::imread(argv[2]);
    faceExchange(srcImage, destImage);

    cv::waitKey(0);
    return 0;
}
