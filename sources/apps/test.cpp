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

void test_yolov3_config()
{
	std::string curr_path{ std::filesystem::current_path().string() };
	std::string data_file = curr_path + "/config/coco.data";
	std::string cfg_file = curr_path + "/config/yolov3.cfg";
	darknet::YoloV3Cfg cfg{ data_file, cfg_file, "kFLOAT" };

	std::cout << "BBOXES: " << cfg.BBOXES << std::endl;
	std::cout << "STRIDE_1: " << cfg.STRIDE_1 << std::endl;
	std::cout << "STRIDE_2: " << cfg.STRIDE_2 << std::endl;
	std::cout << "STRIDE_3: " << cfg.STRIDE_3 << std::endl;
	std::cout << "GRID_SIZE_1: " << cfg.GRID_SIZE_1 << std::endl;
	std::cout << "GRID_SIZE_2: " << cfg.GRID_SIZE_2 << std::endl;
	std::cout << "GRID_SIZE_3: " << cfg.GRID_SIZE_3 << std::endl;
	std::cout << "OUTPUT_SIZE_1: " << cfg.OUTPUT_SIZE_1 << std::endl;
	std::cout << "OUTPUT_SIZE_2: " << cfg.OUTPUT_SIZE_2 << std::endl;
	std::cout << "OUTPUT_SIZE_3: " << cfg.OUTPUT_SIZE_3 << std::endl;
	std::cout << "MASK_1: " << std::endl;
	for (auto& c : cfg.MASK_1)
	{
		std::cout << c << ", ";
	}
	std::cout << std::endl;

	std::cout << "MASK_2: " << std::endl;
	for (auto& c : cfg.MASK_2)
	{
		std::cout << c << ", ";
	}
	std::cout << std::endl;

	std::cout << "MASK_3: " << std::endl;
	for (auto& c : cfg.MASK_3)
	{
		std::cout << c << ", ";
	}
	std::cout << std::endl;



	std::cout << "OUTPUT_BLOB_NAME_1: " << cfg.OUTPUT_BLOB_NAME_1 << std::endl;
	std::cout << "OUTPUT_BLOB_NAME_2: " << cfg.OUTPUT_BLOB_NAME_2 << std::endl;
	std::cout << "OUTPUT_BLOB_NAME_3: " << cfg.OUTPUT_BLOB_NAME_3 << std::endl;


	std::cout << "ANCHORS: " << std::endl;
	for (auto& c : cfg.ANCHORS)
	{
		std::cout << c << ", ";
	}
	std::cout << std::endl;
}

int main()
{
	namespace fs = std::filesystem;
	std::cout << "Current path is " << fs::current_path() << '\n';
	test_parse_config();
	test_yolov3_config();
	return 0;
}
