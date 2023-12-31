#include "general.h"

void test() 
{
    printf("WHOAMI\n");
}

void sys_error(const char *str) 
{
    perror(str);
    exit(1);
}

/**
 * @Author: weiyutao
 * @Date: 2023-01-10 12:09:46
 * @Parameters: 
 * @Return: 
 * @Description: show many image in one window, it means you should define the param
 * that you want to used show type, just like, you should define the min or max width and
 * height if you show different number image. we defined the different min and max width and height
 * when you pass the different numbers image.
 * the max numbers image you can show is 12
 * 1-4: 
 *      1 360*600   1,1
 *      2 360*600   1,2
 *      3 360*600   2,2
 *      4 360*600   2,2
 * 5-9:
 *      5 180*300    2*3
 *      6 180*300    2*3
 *      7 180*300    3*3
 *      8 180*300    3*3
 *      9 180*300    3*3
 * 10-12, the max width and height is 150px
 *      10 90*150    4*3
 *      11 90*150    4*3
 *      12 90*150    4*3
 * just like, if one picture is 200*600, if the max width and height is 300.
 * the you should scale the size of image to 100*300. the height and width can not more than the max height and width.
 * 
 * then, we have define the fixed parma that the function display multi image.
 * if you want to show multi image, you will get the fixed size based on the numbers of the input imagee.
 */
void imshowMulti(string &str, vector<cv::Mat> vectorImage) 
{
    int numImage = (int)vectorImage.size();
    int w, h; // w means the image numbers for row, h means the image numbers for columns.
    // just like, w is 2 if you want to show two image in one window.
    // w is 1 if you want to show one image in one window.
    int height, width; // the height, width that each image based on the numbers of input image.

    if (numImage <= 0)
    {
        printf("the image numbers arguments you passed too small!");
        return;
    }
    else if (numImage > 12)
    {
        printf("the image number arguments you passed too large!");
        return;
    }

    if (numImage <= 4)
    {
        height = 360; width = 600;
        switch (numImage)
        {
        case 1:
            h = 1; w = 1;
            break;
        case 2:
            h = 1; w = 2;
            break;
        default:
            h = 2; w = 2;
            break;
        }
    }
    else if (numImage >= 5 && numImage <= 9)
    {
        height = 180; width = 300;
        switch (numImage)
        {
        case 5:
            h = 2; w = 3;
            break;
        case 6:
            h = 2; w = 3;
            break;
        default:
            h = 3; w = 3;
            break;
        }
    }
    else
    {
        height = 90; width = 150;
        h = 4; w = 3;
    }

    cv::Mat dstImage = cv::Mat::zeros(60 + height * h, 90 + width * w, CV_8UC1);
    // notice, you should start from 20,20. because you should reserved space between two image.
    // m, n is cooresponding the element corrdinates x, y in the dstImage.
    // this bigImage is the output image than involved all input image.
    for (int i = 0, m = 20, n = 10; i < numImage; i++, m += (10 + width))
    {
        if (i % w == 0 && m != 20)
        {
            // if true, you should start from 20, because it must be the right of the window.
            m = 20;
            n += 10 + height;
        }
        // frame of a region in original image dstImage.
        // this region used variable imgROI to show.
        cv::Mat imgROI = dstImage(cv::Rect(m, n, width, height));
        // notice. the first param of Size is width, the second param is height.
        cv::resize(vectorImage[i], imgROI, cv::Size(width, height));
    }
    imshow(str, dstImage);
}

/**
 * @Author: weiyutao
 * @Date: 2023-02-19 21:50:39
 * @Parameters: string dir. fileNames, filePaths. countImage.
 * @Return: .jpg or .png files. and return the numbers of these files.
 * @Description: return all image files from one directory. notice, it can only read files, can not
 * read directory.
 */
void getImageFileFromDir(const string dirPath, std::vector<cv::String> &imageNames, std::vector<cv::String> &imagePaths, int &countImage)
{
    cv::glob(dirPath, imagePaths);
    imageNames = imagePaths;
    for (size_t i = 0; i < imagePaths.size(); i++)
    {
        if ((imagePaths[i].find(".jpg") != imagePaths[i].npos) || (imagePaths[i].find(".png") != imagePaths[i].npos))
        {
            size_t position = imagePaths[i].find_last_of('\\');
            size_t length = imagePaths[i].find_last_of('.');
            imageNames[i] = imagePaths[i].substr(position + 1, length - position - 1);
            countImage++;
        }
    }
}

cv::Mat resizeImage(cv::Mat &inputImage, float scale) 
{
    cv::Size size_ = cv::Size(inputImage.cols * scale, inputImage.rows * scale);
    cv::Mat outputImage = cv::Mat(size_, CV_32S);
    cv::resize(inputImage, outputImage, size_);
    return outputImage;
}

void storeImage(cv::Mat &image) 
{
    time_t currentTime = time(NULL);
    tm *p = localtime(&currentTime);
    char fileName[100] = {0};
    sprintf(fileName, "D:\\development_code_2023-03-03\\vscode\\company\\govern_vue\\src\\assets\\resources\\%d%02d%02d%02d%02d%02d.png", p->tm_year + 1900, p->tm_mon + 1, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
    cv::imwrite(fileName, image);
    fileName[100] = {0};
    sprintf(fileName, "%d%02d%02d%02d%02d%02d.png", p->tm_year + 1900, p->tm_mon + 1, p->tm_mday, p->tm_hour, p->tm_min, p->tm_sec);
    cout << fileName << endl;
}