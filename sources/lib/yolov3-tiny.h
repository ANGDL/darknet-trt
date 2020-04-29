#ifndef _YOLO_V3_TINY_H_
#define _YOLO_V3_TINY_H_

#include "yolo.h"

namespace darknet {
	class YoloV3Tiny : public Yolo
	{
	public:
		explicit YoloV3Tiny(NetConfig* config, uint batch_size);
		void infer(const unsigned char* input);
		std::vector<BBoxInfo> get_detecions(const int image_idx, const int image_w, const int image_h);

		YoloV3TinyCfg* net_cfg;

	private:
		Tensor2BBoxes convert_to_bboxes;

		int output_index_1;
		int output_index_2;
	};
}


#endif
