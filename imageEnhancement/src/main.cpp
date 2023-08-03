/**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-03-29 10:12:35
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-03-29 10:12:35
 * @Description: this process should involved all the operation about the image enhancement.
 * just like the contrast enhancement, drop water mark (involved any type water mark, involved
 * the low gray value water mark and high gray value water mark, you can use the simple method
 * just like threshold function to drop the water mark if you the gray value can be clearly
 * distinguished. of course, you should use the other complexed method to drop the mark if the 
 * it is not clear distinguished, you can specific the region or other.)
***********************************************************************/

#include "enhance.h"


int main(int argc, const char *argv[])
{
    // argc means the input param numbers.
    // notice the first param is the exe file.
    cv::Mat image = cv::imread(argv[1], 0);
    cv::Mat outImage;
    histogramEqualizeTransformation(image, outImage);
    time_t currentTime = time(NULL);
    tm *p = localtime(&currentTime);
    char fileName[100] = {0};
    sprintf(fileName, "D:\\development_code_2023-03-03\\vscode\\company\\govern_vue\\src\\assets\\resources\\%d%02d%02d%02d%02d%02d.png", p->tm_year + 1900, p->tm_mon + 1, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
    imwrite(fileName, outImage);
    fileName[100] = {0};
    sprintf(fileName, "%d%02d%02d%02d%02d%02d.png", p->tm_year + 1900, p->tm_mon + 1, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
    cout << fileName << endl;
    return 0;
}
