#ifndef _DARKNET_CFG_H_
#define _DARKNET_CFG_H_

#include <string>
#include <vector>
#include <map>

namespace darknet {
    using namespace std;

    typedef unsigned int uint;
    typedef vector<map<string, string>> Blocks;
    typedef map<string, string> Block;

    const vector<string> YOLOV3_TINT_OUTPUT_NAMES{"yolo_17", "yolo_24"};
    const vector<string> YOLOV3_OUTPUT_NAMES{"yolo_83", "yolo_95", "yolo_107"};

    class NetConfig {
    public:
        NetConfig(
                const string& data_file,
                const string& yolo_cfg_file,
                string  weights_file,
                string  precision,
                string  input_blob_name = "data");

        void display_blocks();

        bool good() const;

        virtual string get_network_type() const;

        virtual uint get_bboxes() const;

    protected:
        uint init_output_classes(const string& data_file);

        vector<string> init_classes_names(const string& data_file);

        Blocks parse_config2blocks(const string& yolo_cfg_file);

        string find_net_property(const string& property, string default_value);

        bool is_good;
    public:
        const Blocks blocks;

        const string PRECISION;
        const string INPUT_BLOB_NAME;
        const uint INPUT_W;
        const uint INPUT_H;
        const uint INPUT_C;
        const uint INPUT_SIZE;
        const uint OUTPUT_CLASSES;
        const vector<string> CLASS_NAMES;

        const string WEIGHTS_FLIE;

        // custom config
        float nms_thresh_ = 0.6;
        float score_thresh_ = 0.5;
        int max_detection_ = 100;
        bool use_cuda_nms_ = true;
        bool use_dla = false;

        // INT8 setting
        std::string calib_images_list_file;
        std::string calib_images_path;
        std::string calib_table_file_path;
    };

    class YoloV3TinyCfg : public NetConfig {
    public:
        YoloV3TinyCfg(
                const string& data_file,
                const string& yolo_cfg_file,
                const string& weights_file,
                const string& precision,
                const string& input_blob_name = "data",
                const vector<string>& output_names = YOLOV3_TINT_OUTPUT_NAMES);

        virtual string get_network_type() const;

        virtual unsigned int get_bboxes() const;

    protected:
        virtual std::vector<int> find_mask(int idx);

        virtual std::vector<float> find_anchors();

    public:
        const uint BBOXES;
        const uint STRIDE_1;
        const uint STRIDE_2;
        const uint GRID_SIZE_1;
        const uint GRID_SIZE_2;
        const uint OUTPUT_SIZE_1;
        const uint OUTPUT_SIZE_2;
        const std::vector<int> MASK_1;
        const std::vector<int> MASK_2;
        const std::string OUTPUT_BLOB_NAME_1;
        const std::string OUTPUT_BLOB_NAME_2;
        const std::vector<float> ANCHORS;
    };

    class YoloV3Cfg : public YoloV3TinyCfg {
    public:
        YoloV3Cfg(const string& data_file,
                  const string& yolo_cfg_file,
                  const string& weights_file,
                  const string& precision,
                  const string& input_blob_name = "data",
                  const vector<string>& output_names = YOLOV3_OUTPUT_NAMES);

        virtual string get_network_type() const;

    public:
        const uint STRIDE_3;
        const uint GRID_SIZE_3;
        const uint OUTPUT_SIZE_3;
        const std::vector<int> MASK_3;
        const std::string OUTPUT_BLOB_NAME_3;
    };


    class DarkNetCfgFactory {
    public:
        static NetConfig *create_network_config(
                const string& network_type,
                const string& data_file,
                const string& yolo_cfg_file,
                const string& weights_file,
                const string& precision
        );
    };

}

#endif
