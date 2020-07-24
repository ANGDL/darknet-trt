#include <iostream>
#include "../lib/darknet_cfg.h"
#include "../lib/yolov3-nms.h"
#include "opencv2/opencv.hpp"

#include <chrono>


void draw_img(const std::vector<BBoxInfo> &result, cv::Mat &img, const std::vector<cv::Scalar> &color,
              std::vector<std::string> class_names) {
    int box_think = (img.rows + img.cols) * .001;
    float label_scale = img.rows * 0.0009;
    int base_line;
    for (const auto &box : result) {
        const int x = box.box.x1;
        const int y = box.box.y1;
        const int w = box.box.x2 - box.box.x1;
        const int h = box.box.y2 - box.box.y1;

        std::string label;
        std::stringstream stream;
        stream << class_names[box.label] << " " << box.prob << std::endl;
        std::getline(stream, label);

        cv::rectangle(img, cv::Rect(x, y, w, h), color[box.label], box_think);

        auto tsize = cv::getTextSize(label, cv::FONT_HERSHEY_COMPLEX, label_scale, 1, &base_line);

        cv::rectangle(img, cv::Rect(x, y, tsize.width + 3, tsize.height + 4), color[box.label], -1);
        cv::putText(img, label, cv::Point(x, y + tsize.height),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255), box_think / 3, 8, 0);
    }
    cv::imwrite("result.jpg", img);
    cv::imshow("result", img);
    cv::waitKey(0);
}

cv::Mat blob_from_mats(const std::vector<cv::Mat> &input_images, const int &input_w,
                       const int &input_h) {
    std::vector<cv::Mat> letterboxStack(input_images.size());
    for (uint i = 0; i < input_images.size(); ++i) {
        input_images.at(i).copyTo(letterboxStack.at(i));
    }
    return cv::dnn::blobFromImages(letterboxStack, 1.0, cv::Size(input_w, input_h),
                                   cv::Scalar(0.0, 0.0, 0.0), true, false);
}


cv::Scalar randomColor(cv::RNG rng) {
    return cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
}


void test_yolov3nms_infer(const std::string &net_type) {
    const int batch_size = 1;

    int d = 0;
    cudaGetDevice(&d);
//    bool fp16 = d == 2;
    bool fp16 = true;
    std::string curr_path{"."};

    //std::string calib_table_file = "";
    //darknet::NetConfig* cfg = darknet::DarkNetCfgFactory::create_network_config("yolov3", data_file, cfg_file, weights_file, calib_table_file, "kHALF");

    std::string data_file = curr_path + "/../config/coco.data";
    std::string cfg_file = curr_path + "/../config/" + net_type + ".cfg";
    std::string weights_file = curr_path + "/../data/" + net_type + ".weights";
    std::string calib_table_file = "";
    darknet::NetConfig *cfg = darknet::DarkNetCfgFactory::create_network_config(net_type, data_file, cfg_file,
                                                                                weights_file, calib_table_file,
                                                                                fp16 ? "kHALF" : "kFLOAT");

    cfg->use_cuda_nms_ = true;
    cfg->score_thresh_ = 0.6;
    cfg->nms_thresh_ = 0.1;
    darknet::YoloV3NMS net(cfg, batch_size);

    std::vector<cv::Scalar> color;
    for (int i = 0; i < cfg->OUTPUT_CLASSES; ++i) color.push_back(randomColor(cv::RNG(cv::getTickCount())));

    std::vector<cv::Mat> frames;

    cv::Mat frame = cv::imread("person.jpg");
    //cv::imshow("frame", frame);
    //cv::waitKey(0);

    cv::Mat input_mat;

    for (int i = 0; i < batch_size; ++i) {
        frames.push_back(frame);
    }

    input_mat = blob_from_mats(frames, cfg->INPUT_W, cfg->INPUT_H);

    auto start = std::chrono::system_clock::now();

    net.infer(input_mat.data);
    auto bboxes = net.get_detecions(frame.cols, frame.rows);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time to fill and iterate a vector of " << " ints : " << diff.count() << " s\n";

    for (int i = 0; i < frames.size(); ++i) {
        draw_img(bboxes[i], frames[i], color, cfg->CLASS_NAMES);
    }
}

int main(int argc, char **argv) {
    int d = 1;
    cudaGetDeviceCount(&d);
    if (d < 2)
        cudaSetDevice(0);
    else
        cudaSetDevice(2);

    std::string net_type = "yolov3-tiny";
    for (int i = 1; i != argc; ++i) {
        if (!strcmp(argv[i], "-net_type")) {
            if (++i == argc) {
                std::cout << "input error" << std::endl;
            }
            net_type = std::string(argv[i]);
            continue;
        }

    }

    test_yolov3nms_infer(net_type);
    return 0;
}
