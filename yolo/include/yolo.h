/**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-03-27 12:51:07
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-03-27 12:51:07
 * @Description: you can define the head file that is dedicated to using in the current module.
 * you can also define the general head file in the general head module what you need not to compiler
 * the exe file for them. you will just generate the corresponding lib for them. and then you can 
 * reference the lib in the other module what you want to generate the exe file for them.
 * notice, you can not have the same head file name. but you can have the same cpp files in different module in
 * this program.
***********************************************************************/
#ifndef _YOLO_H
#define _YOLO_H

#include "general.h"
#include "general1.h"
#include "netConfig.h"

class YOLO
{
	public:
		YOLO(NetConfig netConfig, bool is_cuda);
		void detect(cv::Mat &inputImage, std::vector<Detection> &outs);
		static vector<string> load_class_list(string classFile);
		static void load_net(cv::dnn::Net &net, string netName, bool is_cuda);
		static cv::Mat format_yolov5(const cv::Mat &inputImage);
		void drawPredict(cv::Mat &frame, std::vector<Detection> &outs);
		vector<string> getClasses();
		~YOLO();
	private:
		string classesFile;
		float inputWidth;
		float inputHeight;

		float confThreshold;
		float iouThreshold;

		float objThreshold;
		
		string netName;
		vector<string> classes;
		cv::dnn::Net net;

		std::vector<cv::Scalar> colors;
};


#endif