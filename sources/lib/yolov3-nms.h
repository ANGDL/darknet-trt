#ifndef _YOLOV3_NMS_H_
#define _YOLOV3__NMS_H_

#include "yolo.h"

namespace darknet {
	class YoloV3NMS : public Yolo
	{
	public:
		explicit YoloV3NMS(NetConfig* config, uint batch_size);
		void infer(const unsigned char* input);
		std::vector<std::vector<BBoxInfo>> get_detecions(const int image_w, const int image_h);

	private:

		int output_index_1;
		int output_index_2;
		int output_index_3;
	};
}

#endif
