#ifndef _DARKNET_CFG_H_
#define _DARKNET_CFG_H_

#include <string>
#include <vector>
#include <map>

namespace darknet {
	using namespace std;

	typedef vector<map<string, string>> Blocks;
	typedef map<string, string> Block;

	class NetConfig {
	public:
		NetConfig(string data_file, string yolo_cfg_file, string precision, string input_blob_name = "data");
		void display_blocks();
		bool good() const;
	private:
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
}

#endif
