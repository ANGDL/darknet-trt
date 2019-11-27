#include "../lib/darknet_cfg.h"
#include <iostream>
#include <filesystem>

void test_parse_config()
{
	std::string curr_path{ std::filesystem::current_path().string() };
	std::string data_file = curr_path + "/config/coco.data";
	std::string cfg_file = curr_path + "/config/yolov3.cfg";
	darknet::NetConfig cfg{ data_file, cfg_file, "kFLOAT" };
	cfg.display_blocks();
	std::cout << "PRECISION: " << cfg.PRECISION << std::endl;
	std::cout << "INPUT_BLOB_NAME: " << cfg.INPUT_BLOB_NAME << std::endl;
	std::cout << "INPUT_W: " << cfg.INPUT_W << std::endl;
	std::cout << "INPUT_H: " << cfg.INPUT_H << std::endl;
	std::cout << "INPUT_C: " << cfg.INPUT_C << std::endl;
	std::cout << "INPUT_SIZE: " << cfg.INPUT_SIZE << std::endl;
	std::cout << "OUTPUT_CLASSES: " << cfg.OUTPUT_CLASSES << std::endl;
	std::cout << "CLASS_NAMES: " << std::endl;
	for (auto& c : cfg.CLASS_NAMES)
	{
		std::cout << c << std::endl;
	}
}

int main()
{
	namespace fs = std::filesystem;
	std::cout << "Current path is " << fs::current_path() << '\n';
	test_parse_config();
	return 0;
}
