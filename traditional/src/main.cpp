#include "traditional.h"



int main(int argc, char const *argv[])
{

    if (argc != 7)
    {
        cout << "====================================================================" << endl;
        cout << "first param is the exe file." << endl;
        cout << "second param is the input image path." << endl;
        cout << "third and fourth param is the orginal coordinate(x, y)." << endl;
        cout << "fifth and sixth param is the width and height of the interest region." << endl;
        cout << "seventh param is the output image path, and your output image path must end with jpg or png." << endl;
        cout << "====================================================================" << endl;
        sys_error("please input at least 7 parameters.\n");
    }
    cv::Mat inputImage = cv::imread(argv[1]);
    // cv::Point pt1(*argv[2], *argv[3]);
    int x = std::stoi(argv[2]);
    int y = std::stoi(argv[3]);
    int width = std::stoi(argv[4]);
    int height = std::stoi(argv[5]);

    cv::Rect rect(x, y, width, height);
    thresholdAndLimitRange(inputImage, rect);

    cv::imshow("WHOAMI", inputImage);
    cv::imwrite(argv[6], inputImage);
    cout << "result image has stored at " << argv[6] << endl;
    cv::waitKey(0);
    system("pause");
    return 0;
}
