#include "darknet_utils.h"


bool file_exits(const std::string filename)
{
	return std::filesystem::exists(std::filesystem::path(filename));
}

static void leftTrim(std::string& s)
{
	s.erase(s.begin(), find_if(s.begin(), s.end(), [](int ch) { return !isspace(ch); }));
}

static void rightTrim(std::string& s)
{
	s.erase(find_if(s.rbegin(), s.rend(), [](int ch) { return !isspace(ch); }).base(), s.end());
}

std::string trim(std::string s)
{
	leftTrim(s);
	rightTrim(s);
	return s;
}

std::vector<std::string> split(const std::string& s, char delimiter)
{
	std::vector<std::string> tokens;
	std::string token;
	std::istringstream tokenStream(s);
	while (std::getline(tokenStream, token, delimiter))
	{
		tokens.push_back(token);
	}
	return tokens;
}

std::vector<float> load_weights(const std::string weights_path, const std::string network_type)
{
	std::vector<float> weights;

	std::ifstream file(weights_path, std::fstream::in);
	if (!file.good())
	{
		std::cout << "open weight file failed !" << std::endl;
		return weights;
	}

	if (!(network_type == "yolov3" || network_type == "yolov3-tiny")) {
		std::cout << "Invalid network type" << std::endl;
		assert(0);
		return weights;
	}

	// Remove 5 int32 bytes of data from the stream belonging to the header
	file.ignore(4 * 5);

	char float_weight[4];

	while (!file.eof())
	{
		file.read(float_weight, 4);
		assert(file.gcount() == 4);
		weights.push_back(*reinterpret_cast<float*> (float_weight));
		if (file.peek() == std::istream::traits_type::eof()) break;
	}

	std::cout << "Total Number of weights read : " << weights.size() << std::endl;
	std::cout << "Loading complete!" << std::endl;

	return weights;
}
