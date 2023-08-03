#ifndef _GENERAL_H
#define _GENERAL_H

#include <iostream>
#include <cstdarg>
#include <string.h>
#include <vector>
#include <map>
#include <sys/types.h>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <thread>
#include <time.h>
#include <stdlib.h>

#include <opencv2/opencv_modules.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/face/facerec.hpp>
#include <vector>
#include <string>

// using namespace cv;
using namespace std;
// using namespace dnn;

void test();
void sys_error(const char *str);
void imshowMulti(string &str, vector<cv::Mat> vectorImage);
void getImageFileFromDir(const string dir, std::vector<cv::String> &imageNames, std::vector<cv::String> &imagePaths, int &countImage);
cv::Mat resizeImage(cv::Mat &inputImage, float scale);
void storeImage(cv::Mat &image);
#endif