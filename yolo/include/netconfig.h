/**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-03-27 14:17:59
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-03-27 14:17:59
 * @Description: some settings attribution for the yolo, involved the confidence threshold, 
 * the non-maximum supression threshold what is the rate of the intersection and the 
 * and set about two objects we have detected. notice, these two objects must be overlapping.
 * or we will show these two objects. the iouThreshold and confThreshold will influence the 
 * detected result. the bigger confThreshold the bigger accuracy about the detected result.
 * and the less objects will be detected. if the iouThreshold is equal to 1. it means we will show
 * these two overlapping object at the same time only when the iouThreshold is greater than 1.  it means
 * we will just show the best accuracy object from these two overlapping objects unless the rate of
 * the intersection and the and set about these two object is greater than 1. but the largest of iou is equal to
 * 1 in fact, so it means we will just show the best accuray object if the overlapping is exists. so generally we should
 * adjust this iouThreshold value a little bit small.
***********************************************************************/
#ifndef _NETCONFIG_H
#define _NETCONFIG_H
#include "general.h"

// define one struct to store the different net attribution for the yolo model.
typedef struct NetConfig
{
    // some attribution about the detected parameters.
    float confThreshold;
    float iouThreshold;

    float objThreshold;

    float inputWidth;
    float inputHeight;

    // the path of file that store the classes, model configuration, model weights and net name.
    string classesFile;
    string netName;
} NetConfigClass, *NetConfigClassP;

typedef struct Detected
{
    int class_id;
    float confidence;
    cv::Rect box;
} Detection, *DetectionP;

// statement one share varilabe. notice you should define this varilabel in another implement files.
extern NetConfigClass yoloNets[4];
#endif