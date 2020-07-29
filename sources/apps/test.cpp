#include <iostream>
#include "../lib/darknet_cfg.h"
#include "../lib/yolov3-nms.h"
#include "opencv2/opencv.hpp"

#include <chrono>
#include <string>


cv::Mat blob_from_mats(const std::vector<cv::Mat> &input_images, const int &input_w,
                       const int &input_h) {
    std::vector<cv::Mat> letterboxStack(input_images.size());
    for (uint i = 0; i < input_images.size(); ++i) {
        input_images.at(i).copyTo(letterboxStack.at(i));
    }
    return cv::dnn::blobFromImages(letterboxStack, 1.0, cv::Size(input_w, input_h),
                                   cv::Scalar(0.0, 0.0, 0.0), true, false);
}

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


cv::Scalar randomColor(cv::RNG rng) {
    return cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
}


void test_yolov3nms_infer(const std::string &net_type, const int batch_size, const int p=0) {
    bool using_opencv = false;

    std::string curr_path{"."};

    const std::vector<std::string> precision{"kFLOAT", "kHALF", "KINT8"};

    std::string data_file = curr_path + "/../config/coco.data";
    std::string cfg_file = curr_path + "/../config/" + net_type + ".cfg";
    std::string weights_file = curr_path + "/../data/" + net_type + ".weights";

    darknet::NetConfig *cfg = darknet::DarkNetCfgFactory::create_network_config(
            net_type, data_file, cfg_file, weights_file,precision[p]);

    cfg->calib_images_list_file = curr_path + "/../data/calib_image_list.txt";
    cfg->calib_table_file_path = curr_path + "/../data/" + net_type + "-coco-calibration.table";

    cfg->use_cuda_nms_ = true;
    cfg->score_thresh_ = 0.6;
    cfg->nms_thresh_ = 0.5;
    darknet::YoloV3NMS net(cfg, batch_size);

    std::vector<cv::Scalar> color;
    for (int i = 0; i < cfg->OUTPUT_CLASSES; ++i) color.push_back(randomColor(cv::RNG(cv::getTickCount())));

    unsigned char* input_data = nullptr;

    if(using_opencv){
        std::vector<cv::Mat> frames;
        cv::Mat frame = cv::imread("person.jpg");
        for (int i = 0; i < batch_size; ++i) {
            frames.push_back(frame);
        }
        auto input_mat = blob_from_mats(frames, cfg->INPUT_W, cfg->INPUT_H);
        input_data = (unsigned char*)(input_mat.data);

        cv::Mat m(cfg->INPUT_H, cfg->INPUT_W, CV_32FC3, input_data);
        cv::cvtColor(m, m, cv::COLOR_RGB2BGR);
        cv::imshow("", m);
        cv::waitKey(0);


        auto start = std::chrono::system_clock::now();

        net.infer(input_data);

        auto bboxes = net.get_detecions(frame.cols, frame.rows);

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Time " << " : " << diff.count() << " s\n";

        for (int i = 0; i < frames.size(); ++i) {
            draw_img(bboxes[i], frames[i], color, cfg->CLASS_NAMES);
        }

    }
    else{
        std::vector<Img> frames;
        Img frame = read_image("person.jpg");

        for (int i = 0; i < batch_size; ++i) {
            frames.push_back(frame);
        }
        auto input_mat = blob_from_images(frames, cfg->INPUT_W, cfg->INPUT_H);
        input_data = (unsigned char*)(input_mat.data.get());

        auto start = std::chrono::system_clock::now();

        net.infer(input_data);
        auto bboxes = net.get_detecions(frame.w, frame.h);

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Time " << " : " << diff.count() << " s\n";

        std::vector<cv::Mat> mats;
        for (auto & i : frames) {
            cv::Mat m(frame.h, frame.w, CV_32FC3, i.data.get());
            m.convertTo(m, CV_8UC3);
            cv::cvtColor(m, m, cv::COLOR_RGB2BGR);
            mats.push_back(m);
        }

        for (int i = 0; i < mats.size(); ++i) {
            draw_img(bboxes[i], mats[i], color, cfg->CLASS_NAMES);
        }
    }
}

int main(int argc, char **argv) {
    int d = 1;
    int batch_size = 1;
    std::string net_type = "yolov3";
    int p = 0;
    int device_id = 0;

    for (int i = 1; i != argc; ++i) {
        if (!strcmp(argv[i], "-batch_size")) {
            if (++i == argc) {
                std::cout << "batch_size input error" << std::endl;
            }
            batch_size = std::stoi(argv[i]);
            continue;
        }

        if (!strcmp(argv[i], "-device")) {
            if (++i == argc) {
                std::cout << "device input error" << std::endl;
            }
            device_id = std::stoi(argv[i]);
            continue;
        }

        if (!strcmp(argv[i], "-fp16")) {
            p = 1;
            continue;
        }

        if (!strcmp(argv[i], "-int8")){
            p = 2;
            continue;
        }

        if (!strcmp(argv[i], "-net_type")) {
            if (++i == argc) {
                std::cout << "input error" << std::endl;
            }
            net_type = std::string(argv[i]);
            continue;
        }

    }

    cudaSetDevice(device_id);

    test_yolov3nms_infer(net_type, batch_size, p);

    return 0;
}
