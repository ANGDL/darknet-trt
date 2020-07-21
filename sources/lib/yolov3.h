#ifndef _YOLOV3_H_
#define _YOLOV3_H_

#include "yolo.h"

namespace darknet {
    class YoloV3 : public Yolo {
    public:
        explicit YoloV3(NetConfig *config, uint batch_size);

        void infer(const unsigned char *input);

        std::vector<BBoxInfo> get_detecions(const int image_idx, const int image_w, const int image_h);

        YoloV3Cfg *net_cfg;

    private:
        Tensor2BBoxes convert_to_bboxes;

        int output_index_1;
        int output_index_2;
        int output_index_3;
    };
}

#endif
