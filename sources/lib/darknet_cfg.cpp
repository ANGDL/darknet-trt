#include "darknet_cfg.h"
#include "darknet_utils.h"
#include <iostream>
#include <fstream>
#include <assert.h>


darknet::NetConfig::NetConfig(string data_file, string yolo_cfg_file, string precision, string input_blob_name) :
	is_good(true),
	blocks(parse_config2blocks(yolo_cfg_file)),
	PRECISION(precision),
	INPUT_BLOB_NAME(input_blob_name),
	INPUT_W(stoi(find_net_property("width", "416"))),
	INPUT_H(stoi(find_net_property("height", "416"))),
	INPUT_C(stoi(find_net_property("channels", "3"))),
	INPUT_SIZE(INPUT_C* INPUT_W* INPUT_H),
	OUTPUT_CLASSES(init_output_classes(data_file)),
	CLASS_NAMES(init_classes_names(data_file))
{

}

void darknet::NetConfig::display_blocks()
{
	for (size_t i = 0; i != blocks.size(); ++i)
	{
		Block block = blocks[i];
		std::cout << "" << std::endl;
		std::cout << "[ " << block.at("type") << " ]" << std::endl;
		for (auto& c : block)
		{
			if (c.first.compare("type") == 0)
				continue;
			std::cout << c.first << "-->" << c.second << std::endl;
		}
		std::cout << std::endl;
	}
}


bool darknet::NetConfig::good() const
{
	return is_good;
}

unsigned int  darknet::NetConfig::init_output_classes(string data_file)
{
	std::fstream sread(data_file, std::fstream::in);

	if (file_exits(data_file) && sread.good())
	{
		std::string line;
		while (getline(sread, line)) {
			std::string::size_type i;
			if ((i = line.find("classes") != std::string::npos))
			{
				i = line.find('=');
				std::string n_classes = trim(line.substr(i + 1));
				return stoi(n_classes);
			}
		}
	}

	std::cout << "read  data file error!" << std::endl;
	is_good = false;
	return 0;
}

std::vector<std::string> darknet::NetConfig::init_classes_names(string data_file)
{
	std::fstream sread(data_file, std::fstream::in);
	std::vector<std::string> names;
	if (file_exits(data_file) && sread.good())
	{
		std::string line;
		while (getline(sread, line)) {
			std::string::size_type i;
			if ((i = line.find("names") != std::string::npos))
			{
				i = line.find('=');
				std::string names_file = trim(line.substr(i + 1));
				std::fstream name_read(names_file, std::fstream::in);
				if (file_exits(names_file) && sread.good()) {
					while (getline(name_read, line)) {
						string name = trim(line);
						names.push_back(name);
					}

					return names;
				}
				else {
					std::cout << "read  class names file error!" << std::endl;
					break;
				}

			}
		}
	}

	is_good = false;
	return names;
}

darknet::Blocks darknet::NetConfig::parse_config2blocks(string yolo_cfg_file)
{
	std::ifstream file(yolo_cfg_file, std::fstream::in);

	std::string line;
	Blocks blocks;
	Block block;


	if (!(file_exits(yolo_cfg_file) && file.good()))
	{
		is_good = false;
		return blocks;
	}

	while (getline(file, line))
	{
		if (line.size() == 0) continue;
		if (line.front() == '#') continue;
		line = trim(line);
		if (line.front() == '[')
		{
			if (block.size() > 0)
			{
				blocks.push_back(block);
				block.clear();
			}
			std::string key = "type";
			std::string value = trim(line.substr(1, line.size() - 2));
			block.insert(std::pair<std::string, std::string>(key, value));
		}
		else
		{
			size_t cpos = line.find('=');
			std::string key = trim(line.substr(0, cpos));
			std::string value = trim(line.substr(cpos + 1));
			block.insert(std::pair<std::string, std::string>(key, value));
		}
	}

	blocks.push_back(block);

	return blocks;
}

std::string darknet::NetConfig::find_net_property(string property, string default_value)
{
	auto it = find_if(blocks.begin(), blocks.end(), [](const Block& b) {return  b.find("type") != b.end() && b.at("type") == "net"; });
	if (it != blocks.end())
	{
		try
		{
			return it->at(property);
		}
		catch (const std::exception&)
		{
			return default_value;
		}
	}
	std::cout << "find net block error!" << std::endl;
	return default_value;
}

darknet::YoloV3TinyCfg::YoloV3TinyCfg(
	string data_file,
	string yolo_cfg_file,
	string weights_file,
	string calib_table_file,
	string precision,
	string input_blob_name,
	vector<string> output_names) :

	NetConfig(data_file, yolo_cfg_file, precision, input_blob_name),
	BBOXES(3),
	STRIDE_1(32),
	STRIDE_2(16),
	GRID_SIZE_1(INPUT_W / STRIDE_1),
	GRID_SIZE_2(INPUT_W / STRIDE_2),
	OUTPUT_SIZE_1(GRID_SIZE_1* GRID_SIZE_1* BBOXES* (OUTPUT_CLASSES + 5)),
	OUTPUT_SIZE_2(GRID_SIZE_2* GRID_SIZE_2* BBOXES* (OUTPUT_CLASSES + 5)),
	MASK_1(find_mask(1)),
	MASK_2(find_mask(2)),
	OUTPUT_BLOB_NAME_1(output_names[0]),
	OUTPUT_BLOB_NAME_2(output_names[1]),
	ANCHORS(find_anchors()),
	TRAINED_WEIGHTS_PATH(weights_file),
	CALIB_TABLE_PATH(calib_table_file)
{

}


const std::string darknet::YoloV3TinyCfg::get_network_type() const
{
	return "yolov3-tiny";
}

std::vector<int> darknet::YoloV3TinyCfg::find_mask(int idx)
{
	int i = 0;
	vector<int> res;
	for (auto& block : blocks)
	{
		if (block.find("type") != block.end() && block.at("type") == "yolo") {
			if (++i == idx) {
				vector<string> mask = split(trim(block.at("mask")), ',');
				for (auto& c : mask) {
					res.push_back(stoi(c));
				}
			}
		}
	}

	return res;
}


std::vector<float> darknet::YoloV3TinyCfg::find_anchors()
{
	vector<float> res;
	for (auto& block : blocks)
	{
		if (block.find("type") != block.end() && block.at("type") == "yolo") {
			vector<string> anchors = split(trim(block.at("anchors")), ',');
			for (auto& c : anchors) {
				res.push_back(stof(c));
			}
		}
	}

	return res;
}

darknet::YoloV3Cfg::YoloV3Cfg(
	string data_file,
	string yolo_cfg_file,
	string weights_file,
	string calib_table_file,
	string precision,
	string input_blob_name,
	vector<string> output_names) :

	YoloV3TinyCfg(data_file, yolo_cfg_file, weights_file, calib_table_file, precision, input_blob_name, output_names),
	STRIDE_3(8),
	GRID_SIZE_3(INPUT_H / STRIDE_3),
	OUTPUT_SIZE_3(GRID_SIZE_3* GRID_SIZE_3* (OUTPUT_CLASSES + 5)),
	MASK_3(find_mask(3)),
	OUTPUT_BLOB_NAME_3(output_names[2])
{

}

const std::string darknet::YoloV3Cfg::get_network_type() const
{
	return "yolov3";
}
