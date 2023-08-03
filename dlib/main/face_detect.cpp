#include "dlib.h"

int main(int argc, const char *argv[])
{
    if (argc != 3)
	{
		sys_error("please input the image and the parameters, the first param is srcImage, the second param is dest image.\n");
	}
    cv::Mat inputImage = cv::imread(argv[1]);
    cv::Mat outputImage;
    std::vector<dlib::rectangle> faces = faceDetectUsedDlib(inputImage, outputImage, DLIB::HOG);

    // store the outputImage and print the name of image in terminal what java process will
    // get it and store it used json object.
    storeImage(outputImage);

    // print the detected information in terminal what
    // java will get it and store it used json object.
    // then java will return it to before port what will
    // show the information in browse label.
    printFacesInformation(faces);
    
    return 0;
}
