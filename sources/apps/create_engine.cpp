//
// Created by ANG on 2020/7/22.
//
#include <string>
#include <fstream>
#include <cstring>
#include "yolov3-nms.h"
#include "darknet_cfg.h"

using namespace std;


int main(int argc, char **argv) {
    int batch_size = 1;
    std::string net_type = "yolov3";
    bool fp16 = false;
    int device_id = 0;

    for (int i = 1; i != argc; ++i) {
        if (!strcmp(argv[i], "-batch_size")) {
            if (++i == argc) {
                std::cout << "batch_size input error" << std::endl;
            }
            batch_size = stoi(argv[i]);
            continue;
        }

        if (!strcmp(argv[i], "-device")) {
            if (++i == argc) {
                std::cout << "device input error" << std::endl;
            }
            device_id = stoi(argv[i]);
            continue;
        }

        if (!strcmp(argv[i], "-fp16")) {
            fp16 = true;
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

    const string curr_path = ".";

    std::string data_file = curr_path + "/../config/coco.data";
    std::string cfg_file = curr_path + "/../config/" + net_type + ".cfg";
    std::string weights_file = curr_path + "/../data/" + net_type + ".weights";
    darknet::NetConfig *cfg = darknet::DarkNetCfgFactory::create_network_config(
            net_type, data_file, cfg_file, weights_file,fp16 ? "kHALF" : "kFLOAT");

    cfg->use_cuda_nms_ = true;
    cfg->score_thresh_ = 0.5;
    cfg->nms_thresh_ = 0.2;
    darknet::YoloV3NMS net(cfg, batch_size);

    return 0;
}
