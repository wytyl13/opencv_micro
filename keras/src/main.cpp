#include "general.h"

int main(int argc, char const *argv[])
{


    cv::Mat image = cv::imread("../../resource/gyy.png");
    cv::imshow("image", image);
    printf("WHOAMI\n");
    test();
    // YOLO yolo;
    // yolo.detect(image);
    
    cv::waitKey(0);
    cv::destroyAllWindows();

    system("pause");

    return 0;
}
