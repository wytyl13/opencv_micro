#include "dlib.h"


/**
 * @Author: weiyutao
 * @Date: 2023-02-15 19:06:30
 * @Parameters: 
 * @Return: 
 * @Description: we will use the professional image recognition module dlib.
 * because of the namespace of dlib is conficted with vector in std. so you should
 * drop one. you can drop to use the namespace dlib. you need not to load the face classifier, 
 * because dlib has contained them.
 */
std::vector<dlib::rectangle> faceDetectUsedDlib(cv::Mat &inputImage, cv::Mat &outputImage,  int mode) 
{
    outputImage = inputImage.clone();
    // Dlib HoG face detected algorithm what is the most efficient algorithm in cpu. it can detect
    // slight positive face. but the HoG model can not detect the small size face. you can train yourself
    // small size face classifier if you want enhance its efficient. of course, we can
    // use another algorithm that dlib has provided. MMOD dlib_dnn model. it is more efficient.
    // and support to run in the GPU. the default method is HOG algorithm.
    // the mode SHAPE68 can detect 68 features in face.
    // you should detect all the faces from the picture used detector what you have created. 
    // then detected the feature from each face used shapePredict you have defined.
    // of course, you should defined dilib type container to accept these return value.
    // and you should loaded the trained model dilb website provided.

    // if you want to use HOG method to detecet the face in dlib, you can detected it used detector directly.
    // if you want to use the MMOD what is the pre-trained neural net work model, you should load the model
    // for the detector.
    
    dlib::cv_image<dlib::bgr_pixel> image(outputImage);
    std::vector<dlib::rectangle> faces;
    if (mode == DLIB::MMOD)
    {
        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
        dlib::deserialize(DLIBMMODMODELFACEDETECT) >> detector;
        faces = detector(image);
        for (unsigned int i = 0; i < faces.size(); i++)
        {
            cv::rectangle(outputImage, cv::Rect(faces[i].left(), faces[i].top(), \
            faces[i].width(), faces[i].width()), cv::Scalar(0, 0, 255), 2, 8, 0);
        }
        return faces;
    }
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    faces = detector(image);
    if (mode == DLIB::SHAPE68)
    {
        // you should add the extra detected model if you want to detect the 68 features in face.
        // of course, you should detected the face first. and then detected 68 features based on
        // the detected face.
        dlib::shape_predictor shapePredict;
        dlib::deserialize(DLIBFACEFEATUREDETECT) >> shapePredict;
        vector<dlib::full_object_detection> shapes;
        for (size_t i = 0; i < faces.size(); ++i)
        {
            shapes.push_back(shapePredict(image, faces[i]));
        }
        for (size_t j = 0; j < faces.size(); ++j)
        {
            if (!shapes.empty())
            {
                for (int i = 0; i < 68; i++)
                {
                    string text = "" + i;
                    cv::Point center = cv::Point(shapes[j].part(i).x(), shapes[j].part(i).y());
                    cv::circle(outputImage, center, 1, cv::Scalar(0, 255, 0), 2, 8, 0);
                    /* putText(outputImage, text, center, FONT_HERSHEY_SCRIPT_SIMPLEX,\
                        0.5, cv::Scalar(0, 0, 255), 0.001, 8); */
                }
            }
            #if 0
            // the link error happend when added these follow code. but we have linked the library of dlib.
            // so we will try to use another method.
            dlib::image_window window(image);
            window.add_overlay(dlib::render_face_detections(shapes));
            window.wait_until_closed();
            #endif
        }
        return faces;
    }
    for (unsigned int i = 0; i < faces.size(); i++)
    {
        cv::rectangle(outputImage, cv::Rect(faces[i].left(), faces[i].top(), \
        faces[i].width(), faces[i].width()), cv::Scalar(0, 0, 255), 2, 8, 0);
    }
    return faces;
}

/**
 * @Author: weiyutao
 * @Date: 2023-02-17 10:42:49
 * @Parameters: 
 * @Return: 
 * @Description: the resnet mdoel of face recognition in dlib used resnet neural network. the concept of
 * resnet is to improve the feature detected rate than other traditional digital image process algorithm
 * or the original neural network. it means you can get more feature vector about one face and compare
 * it with the dest image, you can get more correct recognition result. then, we will deep describe the face
 * recognition concept.
 * 
 * step1, detected the face. you can reduce your working and improve your correct rate if you can ignore
 * the other interference factors.
 * 
 * step2, accurate to face, detected all features of face as far as possible. more features more accuracy.
 * you can use the professional face feature detected, just like shape_predictor_68_face_landmarks.dat, or
 * dlib_face_recognition_resnet_model_v1.dat. of course, you can also use the feature detected method in opencv.
 * of course, it is generally suitable for the feature detected for all object in one picture, it is not dedicated
 * to detecting the face feature.
 * 
 * step3, you have got the face features, then, you should compare them based on the error of feature vectors
 * between the sample and dest. so we can conclude that the difference between the original face recognition and
 * the neural network. the traditional used some dimension reduction method, just like PCA. so it can enhance 
 * the efficient but low accuracy. but resnet model remain the important information of face. so you can use it
 * to get large accuracy. it means the difference between these two method is feature vectors.
 * 
 * step4, compare the feature vectors, you can image that as euclidean distance between two feature vectors.
 * 
 * we have implemented these steps above in the former function used opencv. we used haar classifier to 
 * detected the face, and calculated the feature vectors based on the convariance matrix. then use PCA method
 * to recognize. then, we will use the resnet method in dlib to implement the face recognition function.
 * 
 * in order to use the trianed successful model resnet in dlib, you should define a data type that
 * suitable for it. we have defined it in genearl head file based on the fixed standard about resnet
 * in dlib.
 */
void faceImageRecognitionUsedDlib(const string dirPath, const string targetImagePath) 
{
    cv::Mat image;
    // this a column vector, float dimension is (0, 1).
    vector<dlib::matrix<float, 0, 1>> featureVectors;  // store all the feature vectors of all detected faces.
    float vector_error[30];
    int count = 0;
    int invalidCount = 0;

    #if 1
    // scan all the picture, and stored them, and record the numbers.
    std::vector<cv::String> fileNames, imagePaths;
    getImageFileFromDir(dirPath, fileNames, imagePaths, count);
    // the default sort megthod is dictionary order, you should define the compare rule function
    // if you want to implement the complex sort rule. pass the compare function into the third function.
    // sort(fileNames.begin(), fileNames.end(), compareVectorString);
    // printVector(fileNames);
    #endif

    #if 1
    // load all model, involved face detected, feature detected and resnet model.
    // use HOG method what is a cascade of face classifier in dlib
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor shapePredict;
    dlib::deserialize(DLIBFACEFEATUREDETECT) >> shapePredict;
    anet_type net;
    dlib::deserialize(DLIBRESNETMODEL) >> net;
    // it is equal to Mat image, it is the image type for dlib.
    #endif

    #if 1
    // this step we can name it as detected face and calculated feature vectors.
    string imagePath; // stored the current index k imagePath.
    std::vector<dlib::rectangle> dest; // stored the detected faces of the current index k image.
    std::vector<dlib::matrix<dlib::rgb_pixel>> faces; // stored all the face image
    dlib::full_object_detection shape; // stored the 68 feature data of the current face.
    dlib::matrix<dlib::rgb_pixel> image_dlib, faceImage; // store the current overall image and the face image what got from the current overall image.
    std::vector<dlib::matrix<float, 0, 1>> faceDescriptors; // store all the feature vectors. from index 0 to current index k.
    // read all picture what we have stored them in imagePath.
    for (int k = 0; k < count; k++)
    {
        imagePath = imagePaths[k];
        dlib::load_image(image_dlib, imagePath); // read image and stored used image_dlib.
        std::vector<dlib::rectangle> dest = detector(image_dlib); // detected face used HOG in dlib
        if (dest.size() < 1)
        {
            std::cout << "handling " << imagePath << ", there is not face, ignored..." << endl;
            invalidCount++;
        }
        else if (dest.size() > 1)
        {
            std::cout << "handling " << imagePath << ", detected many faces, ignored..." << std::endl;
            invalidCount++;
        }
        else
        {
            // size == 1, you can go on working. define the variable to store each face.
            // you should get the face from one image, so you should store it as one image.
            // of course, you can also use auto what is the new feature in cpp11.
            // because you can ensure just has one face, so you just need to define one shape.
            // need not to define a vector to store it.
            shape = shapePredict(image_dlib, dest[0]); // get the feature of face.
            // then, you should get the face region as one image from original image.
            dlib::extract_image_chip(image_dlib, dlib::get_face_chip_details(shape, 150, 0.25), faceImage);
            faces.push_back(move(faceImage));
            faceDescriptors = net(faces);
            featureVectors.push_back(faceDescriptors[k - invalidCount]);
            std::cout << "the vector of picture " << imagePaths[k] << std::endl;
        }
    }
    #endif

    #if 1
    // this step we can name it as recognition. but you should get the feature vector of the target image.
    // notice, the size of face could not be small if you detected faces used dlib library.
    // because the dlib library does not support the detected about small size face.
    cv::Mat targetImageMat = cv::imread(targetImagePath);
    if (targetImageMat.empty())
    {
        sys_error("load Mat error, please chech your code or imagePath...");
    }
    dlib::cv_image<dlib::bgr_pixel> targetImageDlib(targetImageMat);
    dlib::matrix<dlib::rgb_pixel> targetFaceImage;
    std::vector<dlib::matrix<dlib::rgb_pixel>> targetFacesImage;
    // detected all faces in this target image, and intercepted then from target image and 
    // stored them used faces_test variable.
    std::vector<dlib::rectangle> faces_test = detector(targetImageDlib);
    for (auto face_test : faces_test)
    {
        auto shape_test = shapePredict(targetImageDlib, face_test);
        dlib::extract_image_chip(targetImageDlib, dlib::get_face_chip_details(shape_test, 150, 0.25), targetFaceImage);
        targetFacesImage.push_back(move(targetFaceImage));
    }
    // resnet the faces_test variable.
    // notice, the input face image size must be equal to the model anet_type we have defined in head file.
    // or you will get the error: Failing expression was i->nr()==NR && i->nc()==NC.
    // All input images must have 150 rows and 150 columns, but we got one with 159 rows and 159 columns.
    std::vector<dlib::matrix<float, 0, 1>> targetFaceDescriptors = net(targetFacesImage);
    std::cout << "the numbers of face in target image is: "<< targetFaceDescriptors.size() << std::endl;
    #endif

    #if 1
    // recognition. compare these two feature vector.
    cv::Point origin;
    int width = 0;
    std::string text;
    for (size_t i = 0; i < targetFaceDescriptors.size(); i++)
    {
        origin.x = faces_test[i].left();
        origin.y = faces_test[i].top();
        width = faces_test[i].width();
        text = "anybody";
        for (size_t j = 0; j < featureVectors.size(); j++)
        {
            vector_error[j] = (double)dlib::length(targetFaceDescriptors[i] - featureVectors[j]);
            if (vector_error[j] < 0.4)
            {
                text = fileNames[j];
                std::cout << "find:" << fileNames[j] << "," << text << std::endl;
            }
        }
        cv::putText(targetImageMat, text, origin, cv::FONT_HERSHEY_SIMPLEX,\
                0.5, cv::Scalar(255, 0, 0), 2, 8, 0);
        cv::rectangle(targetImageMat, cv::Rect(origin.x, origin.y, width, width), cv::Scalar(0, 0, 255), 1, 8, 0);
    }
    #endif

    cv::imshow("result image", targetImageMat);
}

void printFacesInformation(std::vector<dlib::rectangle> faces) 
{
    for (size_t i = 0; i < faces.size(); i++)
    {
        if(i == 0)
        {
            cout << "the number detecetd: " << faces.size() << ";";
        }
        cout << i + 1 << ": the leftUpper coordinates is (" << faces[i].left() << ", " << faces[i].top() << "), " << "width and height is (" << faces[i].width() << ", " << faces[i].height() << ");"; 
    }
}

/**
 * @Author: weiyutao
 * @Date: 2023-04-17 13:29:02
 * @Parameters: 
 * @Return: 
 * @Description: face exchange application.
 */
void faceExchange(cv::Mat &srcImage, cv::Mat &destImage) 
{
    // step1: detected the face and got the landmark for the detected face.
    std::vector<cv::Point2f> landmarkSrcImage, landmarkDestImage;
    faceLandMarkDetection(srcImage, landmarkSrcImage);
    faceLandMarkDetection(destImage, landmarkDestImage);

    // step2: get the convex hull what can also be named as edge points
    // based on the landmark we have detected in step1.
    std::vector<cv::Point2f> convexHullSrcImage, convexHullDestImage;
    getConvexHull(landmarkSrcImage, convexHullSrcImage);
    getConvexHull(landmarkDestImage, convexHullDestImage);

    cout << convexHullSrcImage.size() << endl;
    cout << convexHullDestImage.size() << endl;

    // step3: delaunay triangulation what means the internal triangle for the
    // based on the convexHull what means the facial contour.
    // and we can try to draw the internal triangle in the src image and dest image.
    cv::Mat srcWarped = destImage.clone();
    srcImage.convertTo(srcImage, CV_32F);
    srcWarped.convertTo(srcWarped, CV_32F);
    std::vector<correspondens> delaunayTri;
    getDelaunayTriangulationAndAffineTransformation(convexHullSrcImage, \
                convexHullDestImage, srcImage, 
                srcWarped, delaunayTri);
    
    // calculate mask
    std::vector<cv::Point> convexHull8U;
    for(int i = 0; i < convexHullDestImage.size(); ++i)
    {
        cv::Point pt(convexHullDestImage[i].x, convexHullDestImage[i].y);
        convexHull8U.push_back(pt);
    }
    cv::Mat mask = cv::Mat::zeros(destImage.rows, destImage.cols, destImage.depth());
    cv::fillConvexPoly(mask, &convexHull8U[0], convexHull8U.size(), cv::Scalar(255, 255, 255));
    cv::Rect rect = cv::boundingRect(convexHullDestImage);
    cv::Point center = (rect.tl() + rect.br()) / 2;
    srcWarped.convertTo(srcWarped, CV_8UC3);

    cv::Mat outputImage;
    cv::seamlessClone(srcWarped, destImage, mask, center, outputImage, cv::NORMAL_CLONE);
    cv::imshow("out", outputImage);
    cv::waitKey(0);

}

void faceLandMarkDetection(cv::Mat &inputImage, std::vector<cv::Point2f> &landmark) 
{
    // transform the image from cv type to dlib type.
    dlib::cv_image<dlib::bgr_pixel> image(inputImage);
    // detect all the faces in one inputImage.
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    std::vector<dlib::rectangle> faces = detector(image);

    // find the max area face in all faces that we have detected successful.
    dlib::rectangle rect;
    int face_numbers = faces.size();
    if(face_numbers == 0) 
    {
        sys_error("no faces has detected in the inputImage");
    }
    rect = getMaxAreaRectangle(faces);

    // detect the landmark used dlib that based on deep learning model.
    dlib::shape_predictor shapePredict;
    dlib::deserialize(DLIBFACEFEATUREDETECT) >> shapePredict;
    dlib::full_object_detection  shape = shapePredict(image, rect);

    // define the std::vector<cv::Point2f> that stored the landmark in one face.
    for (int i = 0; i < shape.num_parts(); ++i)
    {        
        float x=shape.part(i).x();        
        float y=shape.part(i).y();  
        landmark.push_back(cv::Point2f(x, y));       
    }
}


/**
 * @Author: weiyutao
 * @Date: 2023-04-17 14:09:42
 * @Parameters: 
 * @Return: 
 * @Description: calculate the area of dlib::rectangle.
 */
dlib::rectangle getMaxAreaRectangle(std::vector<dlib::rectangle> faces) { 

    int face_numbers = faces.size();
    int maxArea = 0;
    int area = 0;
    dlib::rectangle rect;
    for(int i = 0; i < face_numbers; i++)
    {
        area = ((faces[i].right() - faces[i].left() + 1) * (faces[i].bottom() - faces[i].top() + 1));
        if(area > maxArea)
        {
            maxArea = area;
            rect = faces[i];
        }
    }
    return  rect;
}

void getConvexHull(std::vector<cv::Point2f> landmark, std::vector<cv::Point2f> &convexHull) 
{
    std::vector<int> hullIndex;
    cv::convexHull(landmark, hullIndex, false, false);
    for(int i = 0; i < hullIndex.size(); i++)
    {
        convexHull.push_back(landmark[hullIndex[i]]);
    }
}

void getDelaunayTriangulationAndAffineTransformation(std::vector<cv::Point2f> &convexHullSrcImage, \
                std::vector<cv::Point2f> &convexHullDestImage, \
                cv::Mat &srcImage, \
                cv::Mat &srcWarped, \
                std::vector<correspondens> &delaunayTri)
{
    cv::Rect rect(0, 0, srcWarped.cols, srcWarped.rows);
    delaunayTriangulation(convexHullDestImage, delaunayTri, rect); 
    for(auto it = delaunayTri.begin(); it != delaunayTri.end(); it++)
    {
        std::vector<cv::Point2f> t1, t2;
        correspondens corpd = *it;
        for(int i = 0; i < 3; ++i)
        {
            t1.push_back(convexHullSrcImage[corpd.index[i]]);
            t2.push_back(convexHullDestImage[corpd.index[i]]);
        }
        warpTriangle(srcImage, srcWarped, t1, t2);
        printf("WHOAMI\n");
    }   

}

void delaunayTriangulation(std::vector<cv::Point2f> &convexHullDestImage, \
				std::vector<correspondens> &delaunayTri, \
				cv::Rect rect){
    cv::Subdiv2D subdiv(rect);
    for(auto it = convexHullDestImage.begin(); it != convexHullDestImage.end(); it++)
    {
        subdiv.insert(*it);
    }
    std:vector<cv::Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    for (size_t i = 0; i < triangleList.size(); ++i)
	{
		printf("WHOAMIdfefjefjefjeo\n");
		std::vector<cv::Point2f> pt;
		correspondens ind;
		cv::Vec6f t = triangleList[i];
		pt.push_back( cv::Point2f(t[0], t[1]) );
		pt.push_back( cv::Point2f(t[2], t[3]) );
		pt.push_back( cv::Point2f(t[4], t[5]) );
		
		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
		{
			int count = 0;
			for (int j = 0; j < 3; ++j)
				for (size_t k = 0; k < convexHullDestImage.size(); k++)
					if (abs(pt[j].x - convexHullDestImage[k].x) < 1.0   &&  std::abs(pt[j].y - convexHullDestImage[k].y) < 1.0)
					{
						ind.index.push_back(k);
						count++;
					}
			if (count == 3)
				delaunayTri.push_back(ind);
		}
	}
}

void warpTriangle(cv::Mat &srcImage, cv::Mat &srcWarped, \
				std::vector<cv::Point2f> &t1, std::vector<cv::Point2f> &t2)
{
    cv::Rect r1 = cv::boundingRect(t1);
    cv::Rect r2 = cv::boundingRect(t2);
    
    // Offset points by left top corner of the respective rectangles
    std::vector<cv::Point2f> t1Rect, t2Rect;
    std::vector<cv::Point> t2RectInt;
    for(int i = 0; i < 3; i++)
    {
        t1Rect.push_back(cv::Point2f(t1[i].x - r1.x, t1[i].y -  r1.y));
        t2Rect.push_back(cv::Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
        t2RectInt.push_back(cv::Point(t2[i].x - r2.x, t2[i].y - r2.y));
    }
    
    // Get mask by filling triangle
    cv::Mat mask = cv::Mat::zeros(r2.height, r2.width, CV_32FC3);
    cv::fillConvexPoly(mask, t2RectInt, cv::Scalar(1.0, 1.0, 1.0), 16, 0);
    
    // Apply warpImage to small rectangular patches
    cv::Mat srcRect;
    srcImage(r1).copyTo(srcRect);
    
    cv::Mat destRect = cv::Mat::zeros(r2.height, r2.width, srcRect.type());
    
    applyAffineTransform(destRect, srcRect, t1Rect, t2Rect);
    
    cv::multiply(destRect, mask, destRect);
    cv::multiply(srcWarped(r2), cv::Scalar(1.0,1.0,1.0) - mask, srcWarped(r2));
    srcWarped(r2) = srcWarped(r2) + destRect;
}

void applyAffineTransform(cv::Mat &destRect, cv::Mat &srcRect, \
                std::vector<cv::Point2f> &t1Rect, std::vector<cv::Point2f> &t2Rect)
{
    // Given a pair of triangles, find the affine transform.
    cv::Mat warpMat = cv::getAffineTransform(t1Rect, t2Rect);
    
    // Apply the Affine Transform just found to the src image
    cv::warpAffine(srcRect, destRect, warpMat, destRect.size(), \
                cv::INTER_LINEAR, cv::BorderTypes::BORDER_REFLECT_101);
}