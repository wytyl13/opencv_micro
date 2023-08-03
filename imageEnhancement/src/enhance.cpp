#include "enhance.h"



/**
 * @Author: weiyutao
 * @Date: 2023-01-25 20:05:40
 * @Parameters: 
 * @Return: 
 * @Description: get the distribution of the inputImage. pass the inputImage, return the 
 * we can use list in cpp, the index of the list is the gray value. the value in the list is the appearance
 * numbers of the gray value in this image. of course, we can also use the map in cpp container.
 * 
 * so we have learned a method, we do not use the static to modify the local variable, 
 * if you want to return the local variabel in one function, you should malloc it or use the reference parameter
 * to receive the content you want to get.
 */
double* getDistribution(const cv::Mat &inputImage)
{
    // it will return the same address if you used the static to modify the array variable.
    // so we will use malloc function to create this variable. and return it.
    // you should define a pointer first, then malloc a address for this pointer in heap.
    // static double list[256] = {0};
    double *list;
    list = (double *)malloc(sizeof(double) * 256);
    // notice you should distinguish the size and size_t, the former is the numbers in array
    // and the last is the memory size waste for the array.
    // you must initialize the array inside the function, or you will get the error numbers.
    memset(list, 0.0, sizeof(double) * 256);
    const uchar *matRow;
    int rows = inputImage.rows;
    int cols = inputImage.cols;
    double MN = rows * cols;
    for (int i = 0; i < rows; i++)
    {
        matRow = inputImage.ptr<uchar>(i);
        int grayValue = 0;
        for (int j = 0; j < cols; j++)
        {
            grayValue = matRow[j];
            list[grayValue] += 1;    
        }
    }
    // int len = sizeof(list) / sizeof(list[0]);
    // the normalized, in order to handle the problem about the size is different between two image.
    for (int i = 0; i < 256; i++)
    {
        list[i] /= MN;
        // rounded and remain two decimal places.
        // list[i] = (int)((list[i] * 100) + 0.5) / 100.0;
        // list[i] = round(list[i] * 100) / 100;
    }
    return list;
}


/**
 * @Author: weiyutao
 * @Date: 2023-01-25 18:20:28
 * @Parameters: 
 * @Return: the image after doing histogram equalize transformation.
 * @Description: histogram equalize transformation processing. 
 * p(rk) = h(rk) / MN = nk / MN
 * M N is the numbers of rows and columns, for all p value of k, sum p(rk) = 1;
 * the component of p(rk) is the probability estimates for the gray value in the image.
 * the histogram is the basic operation in the image processing. because it is simple and 
 * suitable for hardware implementation quickly. so the histogram technology is a popular tool
 * for the real-time image processing. the shape of the histogram is related with the appearance
 * of the image. we can image the relationship between histogram and the appearance of the image.
 * the horizontal axis of the histogram is the gray value. the histogram will foucuse on the left
 * if the picture is the dark image. the histogram will foucuse on the right if the picture is the
 * bright image. the histogram will foucuse on the middle if the picture is low contrast.
 * the histogram will be uniform distribution if the picture is high contrast.
 * so h(rk) is the numbers of the gray value appear in the image.
 * h(rk) / MN is the probabilty of the gray value appear in the image.
 * 
 * the concept of histogram.
 * just like a 3 bit image, L = 2^3 = 8. L - 1 = 7. a 64*64 = 4096 image.
 * we can get the gray scale distribution for this image.
 * gray value       numbers         probability
 * 0                790             790/4096=0.19
 * 1                1023            0.25
 * 2                850             0.21
 * 3                656             0.16
 * 4                329             0.08
 * 5                245             0.06
 * 6                122             0.03
 * 7                81              0.02
 * the normalized image histogram. it is a expression. histogram equalization.
 * s_k = T(r_k) = (L - 1) * Σ(j = 0~k) p_r(r_j)
 * s0 = 7*p_r(r0) = 7 * 0.19 = 1.33 -rounded-> 1 
 * s1 = 7*[p_r(r0)+p_r(r1)] = 7*(0.19+0.25) = 3.08 -rounded->3
 * s2 = 5
 * s3 = 6
 * s4 = 6
 * s5 = 7
 * s6 = 7
 * s7 = 7
 * we can see, the mapping for this expression is one to many mapping.
 * the gray value is monotonous increasing.
 * the mapping of the gray value has two cases.
 * first is monotone increasing function, it means one to many mapping.
 * second is strictly monotone increasing function, it means one to one mapping.
 * it is monotone increasing function in this case.
 * then we can get another hostogram based on the mapping gray value.
 * 1        790                 790         0.19   
 * 3        1023                1023        0.25
 * 5        850                 850         0.21
 * 6        656+329             985         0.24
 * 7        245+122+81          448         0.11
 *
 * the histogram is the approximate of the probabilty density function.
 * this case used normolized image histogram. it will cover the border gray range after
 * the equalization of the image. it means we can enhance the image contrast by using
 * normolized image histogram. this is a specific histogram transform.
 * you can see the probabilty will be balanced for each gray value, and the gray value 
 * will be similar to the original gray value.
 * and a special feature is that, no matter what type image you given, you can get the same high contrast
 * image. it means, if you give some images that the different darker or brightness of one image.
 * carrying out the transformation of histogram equalization to them. you can get the different
 * histogram for the image after transformation, and the result image is euqal.
 * the didfferent darker or brightness image of one same image, you can get the same high contrast
 * image by doing the histogram equalization transformation for them. and the histogram for all high
 * contrast you have got are different. it can also mean that the histogram of the same image may be different.
 * it means the different histogram of one image can show the same contrast picture.
 * why? because all the image just has the different contrast for one same image, so you can get the same
 * high contrast image by using histogram equalize transformation. if this condition do not reach, you can not
 * get this effection.
 * 
 * of course, you can define histogram transform used the official function that opencv has defined.
 * it is equalizeHhist in opencv. 
 * but it may not be appropriate used histogram equalization in some appication. because the histogram equalization
 * will generate the not sure histogram. but sometimes we may want to generate the specific shape histogram.
 * it can be name as histogram matching. the difference between the histogram equalization and histogram
 * matching is the former will generate the unknown histogram based on the known histogram transformation function.
 * the last will generate the known shape histogram based on the changeable histogram transformation function.
 * the former the transformation funcition is the only, the last transformation function need to be calculated
 * according to the known shape of the histogram. the last method can be named the histogram matching.
 * 
 * the histogramEqualizeTransformation function is invalid for the brightness image. because it can just
 * improve the details about the low gray level value. so we should define the histogramMatchingTransformation
 * function.
 */ 
void histogramEqualizeTransformation(const cv::Mat &inputImage, cv::Mat &outputImage) 
{
    // you should get the histogram of the original image.
    double *distribution = getDistribution(inputImage);
    double transformValue[256] = {0};
    // then, transform the distribution.
    double s = 0.0;
    for (int r = 0; r < 256; r++)
    {
        for (int j = 0; j <= r; j++)
        {
            s += distribution[j];
        }
        s *= 255;
        transformValue[r] = round(s);
        s = 0.0;
    }
    // change the original gray value.
    outputImage = inputImage.clone();
    int rows = outputImage.rows;
    int cols = outputImage.cols;
    uchar *rowMat;
    for (int i = 0; i < rows; i++)
    {
        rowMat = outputImage.ptr<uchar>(i);
        for (int j = 0; j < cols; j++)
        {
            rowMat[j] = (uchar)transformValue[rowMat[j]];
        }
    }
}