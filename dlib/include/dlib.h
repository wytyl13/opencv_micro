#ifndef _DLIB_H
#define _DLIB_H

#include "general.h"
#include <dlib/opencv.h>
#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/data_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing/object_detector.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/dnn/core.h>
#include <dlib/dnn/layers.h>

#define OPENCVHAARFACEDETECT "D:/development_app2/opencv/source/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml"
#define OPENCVHAAREYEDETECT "D:/development_app2/opencv/source/opencv/data/haarcascades/haarcascade_eye.xml"
#define OPENCVHAARFACEDETECT_EXTRA "D:/development_app2/opencv/source/opencv/data/haarcascades/haarcascade_frontalface_alt_tree.xml"
#define DLIBMMODMODELFACEDETECT "D:/development_code_2023-03-03/vscode/resources/model/mmod_human_face_detector.dat"
#define DLIBFACEFEATUREDETECT "D:/development_code_2023-03-03/vscode/resources/model/shape_predictor_68_face_landmarks.dat"
#define DLIBRESNETMODEL "D:/development_code_2023-03-03/vscode/resources/model/dlib_face_recognition_resnet_model_v1.dat"


typedef enum DlibFaceDetectAlgorithm
{
    HOG,
    MMOD,
    SHAPE68
} DLIB;


template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;
 
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;
 
template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;
 
template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;
 
using anet_type = dlib::loss_metric<dlib::fc_no_bias<128, dlib::avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<32, 7, 7, 2, 2,
	dlib::input_rgb_image_sized<150>
	>>>>>>>>>>>>;



template <long num_filters, typename SUBNET> using con5d = dlib::con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = dlib::con<num_filters,5,5,1,1,SUBNET>;
 
template <typename SUBNET> using downsampler  = dlib::relu<dlib::affine<con5d<32, dlib::relu<dlib::affine<con5d<32, dlib::relu<dlib::affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = dlib::relu<dlib::affine<con5<45,SUBNET>>>;
 
using net_type = dlib::loss_mmod<dlib::con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>>;


struct correspondens
{
	std::vector<int> index;
};


void func(int, void *);
void getTrainDataFromeDir(const string directoryPath, vector<string> &imagePath);
void getFaceSamplesFromMovie(const string moviePath, const string saveDirectory);

// opencv face detect and recognition.
void faceDetectImage(cv::Mat &inputImage, cv::Mat &ouputImage, cv::CascadeClassifier &face_cascade, cv::CascadeClassifier &eye_cascade);
void faceDetectMovie(const string windowName, const string path, cv::CascadeClassifier &face_cascade, cv::CascadeClassifier &eye_cascade);
void faceRecognition(cv::Mat &inputImage, cv::Mat &outputImage);
void faceRecognitionUsedEigenFace(string labelFile, const string predictMoviePath);

// dlib face detect and recognition.
std::vector<dlib::rectangle> faceDetectUsedDlib(cv::Mat &inputImage, cv::Mat &outputImage, int mode);
void faceImageRecognitionUsedDlib(const string dirPath, const string targetImage);
void faceMovieRecognitionUsedDlib(const string dirPath, const string targetMoivePath);
void printFacesInformation(std::vector<dlib::rectangle> faces);
void faceExchange(cv::Mat &srcImage, cv::Mat &destImage);
void faceLandMarkDetection(cv::Mat &inputImage, std::vector<cv::Point2f> &landmark);
dlib::rectangle getMaxAreaRectangle(std::vector<dlib::rectangle> faces);
void getConvexHull(std::vector<cv::Point2f> landmark, std::vector<cv::Point2f> &convexHull);
void getDelaunayTriangulationAndAffineTransformation(std::vector<cv::Point2f> &convexHullSrcImage, \
                std::vector<cv::Point2f> &convexHullDestImage, \
                cv::Mat &srcImage, \
                cv::Mat &srcWarped, \
                std::vector<correspondens> &delaunayTri);
void delaunayTriangulation(std::vector<cv::Point2f> &convexHullDestImage, \
				std::vector<correspondens> &delaunayTri, \
				cv::Rect rect);
void warpTriangle(cv::Mat &srcImage, cv::Mat &srcWarped, \
				std::vector<cv::Point2f> &t1, std::vector<cv::Point2f> &t2);
void applyAffineTransform(cv::Mat &destRect, cv::Mat &srcRect, \
                std::vector<cv::Point2f> &t1Rect, std::vector<cv::Point2f> &t2Rect);

#endif

