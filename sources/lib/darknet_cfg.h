#ifndef _DARKNET_CFG_H_
#define _DARKNET_CFG_H_

#include <string>
#include <vector>
#include <map>

namespace darknet {
	using namespace std;

	typedef vector<map<string, string>> Blocks;
	typedef map<string, string> Block;

	const vector<string> YOLOV3_TINT_OUTPUT_NAMES{ "yolo_17", "yolo_24" };
	const vector<string> YOLOV3_OUTPUT_NAMES{ "yolo_83", "yolo_95",  "yolo_107" };

	class YoloV3Cfg;

	class NetConfig {
	public:
		NetConfig(string data_file, string yolo_cfg_file, string precision, string input_blob_name = "data");
		void display_blocks();
		bool good() const;
	protected:
		uint32_t init_output_classes(string data_file);
		vector<string> init_classes_names(string data_file);
		Blocks parse_config2blocks(string yolo_cfg_file);
		string find_net_property(string property, string default_value);
		bool is_good;
	public:
		const Blocks blocks;

		const string PRECISION;
		const string INPUT_BLOB_NAME;
		const uint32_t INPUT_W;
		const uint32_t INPUT_H;
		const uint32_t INPUT_C;
		const uint32_t INPUT_SIZE;
		const uint32_t OUTPUT_CLASSES;
		const vector<string> CLASS_NAMES;
	};

	class YoloV3TinyCfg : public NetConfig
	{
	public:
		YoloV3TinyCfg(string data_file,
			string yolo_cfg_file,
			string precision,
			string input_blob_name = "data",
			vector<string> output_names = YOLOV3_TINT_OUTPUT_NAMES);
	protected:
		virtual std::vector<int> find_mask(int idx);
		virtual std::vector<float> find_anchors();

	public:
		const uint32_t BBOXES;
		const uint32_t STRIDE_1;
		const uint32_t STRIDE_2;
		const uint32_t GRID_SIZE_1;
		const uint32_t GRID_SIZE_2;
		const uint32_t OUTPUT_SIZE_1;
		const uint32_t OUTPUT_SIZE_2;
		const std::vector<int> MASK_1;
		const std::vector<int> MASK_2;
		const std::string OUTPUT_BLOB_NAME_1;
		const std::string OUTPUT_BLOB_NAME_2;
		const std::vector<float> ANCHORS;
	};

	class YoloV3Cfg : public YoloV3TinyCfg
	{
	public:
		YoloV3Cfg(string data_file,
			string yolo_cfg_file,
			string precision,
			string input_blob_name = "data",
			vector<string> output_names = YOLOV3_OUTPUT_NAMES);

	public:
		const uint32_t STRIDE_3;
		const uint32_t GRID_SIZE_3;
		const uint32_t OUTPUT_SIZE_3;
		const std::vector<int> MASK_3;
		const std::string OUTPUT_BLOB_NAME_3;
	};
}

#endif
