#include "darknet_cfg.h"
#include "darknet_utils.h"
#include <iostream>
#include <fstream>


darknet::NetConfig::NetConfig(string data_file, string yolo_cfg_file, string precision, string input_blob_name) :
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

uint32_t darknet::NetConfig::init_output_classes(string data_file)
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

	return 20;  // 按voc数据集设置为20
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
					throw std::runtime_error("open " + names_file + "failed!\n");
				}

			}
		}
	}

	names = {
		"aeroplane",
		"bicycle",
		"bird",
		"boat",
		"bottle",
		"bus",
		"car",
		"cat",
		"chair",
		"cow",
		"diningtable",
		"dog",
		"horse",
		"motorbike",
		"person",
		"pottedplant",
		"sheep",
		"sofa",
		"train",
		"tvmonitor"
	};

	return names;
}

darknet::Blocks darknet::NetConfig::parse_config2blocks(string yolo_cfg_file)
{
	std::ifstream file(yolo_cfg_file, std::fstream::in);

	std::string line;
	Blocks blocks;
	Block block;
	if (!(file_exits(yolo_cfg_file) && file.good())) {
		throw std::runtime_error("open " + yolo_cfg_file + "failed!\n");
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
