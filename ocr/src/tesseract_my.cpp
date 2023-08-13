#include<opencv2\opencv.hpp>
#include<iostream>
using namespace cv;
using namespace std;
bool EqualizeHist(Mat gray, Mat result) {
     //统计0~255像素值的个数
     map<int, int>mp;
     for (int i = 0; i < gray.rows; i++){
           uchar* ptr = gray.data + i * gray.cols;
           for (int j = 0; j < gray.cols; j++){
               int value = ptr[j];
               mp[value]++;
           }
      }
      //统计0~255像素值的频率，并计算累计频率
      map<int, double> valuePro;
      int sum = gray.cols*gray.rows;
      double  sumPro = 0;
      for (int i = 0; i < 256; i++) {
          sumPro += 1.0*mp[i] / sum;
          valuePro[i] = sumPro;
      }
      //根据累计频率进行转换
      for (int i = 0; i < gray.rows; i++) {
          uchar* ptr1 = gray.data + i*gray.cols;
          for (int j = 0; j < gray.cols; j++) {
             int value = ptr1[j];
             double p = valuePro[value];
             result.at<uchar>(i, j) = p*value;
          }
       }
       return true;
}


void histogram()
{
    Mat scrImage = imread("c:/users/80521/desktop/gyy.png");
    Mat image = scrImage.clone();
    Mat imageRGB[3];
    split(scrImage, imageRGB);
    for (int i = 0; i < 3; i++) {
        EqualizeHist(imageRGB[i], imageRGB[i]);
    }
    merge(imageRGB, 3, scrImage);
    imshow("原图",image);
    imshow("直方图后的图像", scrImage);
    waitKey(0);
    system("pause");
}

void changeBackground(const Mat& inputImage, const Scalar& backgroundColor, const Scalar& lowerBound, const Scalar& upperBound)
{
 
 
    // 图片转为hsv格式
    Mat hsv;
    cvtColor(inputImage, hsv, COLOR_BGR2HSV);
    // 在指定范围内的变为白色，不在范围内的变为黑色
    Mat mask;
    inRange(hsv, lowerBound, upperBound, mask);
    imwrite("c:/users/80521/desktop/mask.jpg", mask);
 
    // // 取反操作，抠出人像
    // bitwise_not(mask, mask);
 
 
    // 创建新的背景图像
    Mat newBackground = Mat::zeros(inputImage.size(), inputImage.type());
    newBackground = backgroundColor;
 
    // 将原始图像复制到新背景图像中，只保留前景（人像）区域
    inputImage.copyTo(newBackground, mask);
 
    imshow("New Background Image", newBackground);
 
 
    //保存图片
    imwrite("c:/users/80521/desktop/result.jpg", newBackground);
 
 
}

// void change_back()
// {
// // 显示一张图片
//     Mat image = imread("c:/users/80521/desktop/11111.jpg");
//     imshow("1",image);
//     // 检查图像是否成功加载
//     if (image.empty())
//     {
//         cout << "Failed to load image." << endl;
//         return -1;
//     }
 
//     // 定义背景颜色、颜色范围
//     Scalar backgroundColor(219,141, 65);
//     Scalar lowerBound(0, 0, 0);
//     Scalar upperBound(245, 245, 245);
 
//     // 更换背景
//     changeBackground(image, backgroundColor, lowerBound, upperBound);
//     waitKey(0);

//     system("pause");
// }



int main()
{

    Mat image = imread("c:/users/80521/desktop/wgs.jpg");
    Mat newBackground = Mat::zeros(Size(453, 679), image.type());
    // define the subimage and update it will influence the original image.
    Rect subROI_up(0, 0, 453, 100);
    Rect subROI_left(0, 100, 20, 579);
    Rect subROI_right(433, 100, 20, 579);
    Mat subBackground_up = newBackground(subROI_up);
    Mat subBackground_left = newBackground(subROI_left);
    Mat subBackground_right = newBackground(subROI_right);
    subBackground_up.setTo(Scalar(232, 144, 60));
    subBackground_left.setTo(Scalar(232, 144, 60));
    subBackground_right.setTo(Scalar(232, 144, 60));


    // for (int i = 0; i < newBackground.rows; i++) 
    // {
    //     uchar *row_data = newBackground.ptr<uchar>(j);
    //     for (int j = 0; j < newBackground.cols; j++)
    //     {
    //         // 60, 144, 232
    //         row_data[j] = Scalar(60, 144, 232);
    //     }
    // }
    Mat imageROI = newBackground(Rect(20, 100, 413, 579));
    image.copyTo(imageROI);
    imshow("new", newBackground);
    imwrite("c:/users/80521/desktop/result2.jpg", newBackground);
    waitKey(0);
    system("pause");
    return 0;
}