/**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-03-27 12:48:46
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-03-27 12:48:46
 * @Description: because we have add all the include directory of each module in the program.
 * so you should include the different head file in different module. it means you should define
 * the different head file name even though the head file is based on the different module.
***********************************************************************/
#include "yolo.h"



int main(int argc, const char *argv[])
{

	// if (argc != 3)
	// {
	// 	sys_error("please input two parameters.\n");
	// }
	bool is_cuda = false;
	cv::Mat inputImage = cv::imread(argv[1]);
	YOLO yolo_model(yoloNets[0], is_cuda);
	std::vector<Detection> outs;
	yolo_model.detect(inputImage, outs);
	yolo_model.drawPredict(inputImage, outs);
	storeImage(inputImage);
	// transform from vector to list
	// std::list<Detection> lst(outs.begin(), outs.end());
	vector<string> classes = yolo_model.getClasses();
	for (size_t i = 0; i < outs.size(); i++)
	{
		if (i == 0)
		{
			cout << "the number detected is " << outs.size() << ".;";
		}
		cout << classes[outs[i].class_id] << ": " << outs[i].box << ", " << outs[i].confidence << ";";
	}

	// static const string kWinName = "Deep learning object detection in OpenCV";
	// namedWindow(kWinName, 1);
	// imshow(kWinName, inputImage);
	// waitKey(0);
	// destroyAllWindows();
	return 0;
}
