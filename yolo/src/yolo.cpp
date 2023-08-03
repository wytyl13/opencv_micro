#include "yolo.h"

// define the share variable in the implement file.
// you can use extern keyword to define the same variable yoloNets and need not to
// define it in other files. you will get the share variable yoloNets.
NetConfigClass yoloNets[4]
{
  {0.4, 0.4, 0.2, 640.0, 640.0, "D:\\development_code_2023-03-03\\vscode\\opencv_micro\\resource\\model\\classes.txt", "D:\\development_code_2023-03-03\\vscode\\opencv_micro\\resource\\model\\yolov5x.onnx"},
  {0.4, 0.4, 0.2, 460, 460, "config_files/classes.txt", "yolov5n.onnx"}
};

vector<string> YOLO::load_class_list(string classFile) 
{
    std::vector<std::string> class_list;
    std::ifstream ifs(classFile);
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}

void YOLO::load_net(cv::dnn::Net &net, string netName, bool is_cuda) 
{
    auto result = cv::dnn::readNet(netName);
    if (is_cuda)
    {
        // std::cout << "Attempty to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else
    {
        // std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}


cv::Mat YOLO::format_yolov5(const cv::Mat &inputImage) 
{
    int col = inputImage.cols;
    int row = inputImage.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    inputImage.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void YOLO::detect(cv::Mat &inputImage, std::vector<Detection> &outs) 
{
    /*     
    blobFromImage(InputArray image, 
                double scalefactor=1.0, 
                const Size& size = Size(),
                const Scalar& mean = Scalar(), 
                bool swapRB = false, 
                bool crop = false,
                int ddepth = CV_32F) 
    */
    // we can use blobFromImage function, what is a preprocess function involved two steps.
    // first remove the mean based on the image element, then scale the image based on the parameter you have passed.
    // PRPROCESS THE INPUTIMAGE.
    cv::Mat blob;

    auto input_image = format_yolov5(inputImage);
    cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(this->inputWidth, this->inputHeight), cv::Scalar(), true, false);
    this->net.setInput(blob);
    std::vector<cv::Mat> outputs;
    this->net.forward(outputs, this->net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / this->inputWidth;
    float y_factor = input_image.rows / this->inputHeight;

    float *data = (float *)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {

        float confidence = data[4];
        if (confidence >= this->confThreshold) {
            float *classes_scores = data + 5;
            cv::Mat scores(1, this->classes.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > this->objThreshold) {

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }

        }
        data += 85;
    }
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, this->objThreshold, this->iouThreshold, nms_result);
    for (long long unsigned int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        outs.push_back(result);
    }
}

vector<string> YOLO::getClasses() 
{
    return this->classes;
}

void YOLO::drawPredict(cv::Mat &frame, std::vector<Detection> &outs)
{
    int detections = outs.size();
    for (int i = 0; i < detections; ++i)
    {
        auto detection = outs[i];
        auto confidence = detection.confidence;
        auto box = detection.box;
        auto classId = detection.class_id;
        const auto color = this->colors[classId % colors.size()];
        cv::rectangle(frame, box, color, 3);
        cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
        string text = this->classes[classId].c_str();
        text.append(", ");
        text.append(std::to_string(confidence));
        cv::putText(frame, text, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
}

YOLO::YOLO(NetConfig netConfig, bool is_cuda)
{
    this->confThreshold = netConfig.confThreshold;
    this->iouThreshold = netConfig.iouThreshold;

    this->objThreshold = netConfig.objThreshold;

    this->inputWidth = netConfig.inputWidth;
    this->inputHeight = netConfig.inputHeight;

    // you can use the function strcpy_s that dedicated to copying one string to another.
    // you can also use the c_str function to transform one char to string.
    this->netName = netConfig.netName;

    // read the classFile file used stream.
    // and init the classes attribution in class YOLO use the data that read from the classFile
    // attribution in struct NetConfig.
    this->classes = load_class_list(netConfig.classesFile);

    // init the Net attribution in class YOLO. involved the configuration, weights what read from the netConfig parameter,
    // and the train parameters, backend is opencv and version is cpu not the gpu.
    this->load_net(this->net, this->netName, is_cuda);

    this->colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};
}

YOLO::~YOLO()
{
}

