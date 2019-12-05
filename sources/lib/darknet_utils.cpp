#include <functional>
#include <algorithm>
#include "darknet_utils.h"
#include "NvInfer.h"

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

	std::ifstream file(weights_path, std::ios_base::binary);
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

int get_num_channels(nvinfer1::ITensor* t)
{
	nvinfer1::Dims d = t->getDimensions();
	assert(d.nbDims == 3);

	return d.d[0];
}


bool save_engine(const nvinfer1::ICudaEngine* engine, const std::string& file_name)
{
	std::ofstream engineFile(file_name, std::ios::binary);
	if (!engineFile)
	{
		std::cout << "Cannot open engine file: " << file_name << std::endl;
		return false;
	}

	auto serialized_engine = std::unique_ptr < nvinfer1::IHostMemory, std::function<void(nvinfer1::IHostMemory*)>>(
		engine->serialize(),
		[](nvinfer1::IHostMemory* p) {
			p->destroy();
		});

	if (serialized_engine == nullptr)
	{
		std::cout << "Engine serialization failed" << std::endl;
		return false;
	}

	engineFile.write(static_cast<char*>(serialized_engine->data()), serialized_engine->size());


	std::cout << "plan file size: " << serialized_engine->size() << std::endl;

	return !engineFile.fail();
}

nvinfer1::ICudaEngine* load_trt_engine(const std::string plan_file, nvinfer1::IPluginFactory* plugin_factory, nvinfer1::ILogger& logger)
{
	std::cout << "loading trt engine form " << plan_file << std::endl;

	assert(file_exits(plan_file));

	std::ifstream engine_file(plan_file, std::ios::binary);
	if (!engine_file)
	{
		std::cout << "Error opening engine file: " << plan_file << std::endl;
		return nullptr;
	}

	engine_file.seekg(0, engine_file.end);
	long int fsize = engine_file.tellg();
	engine_file.seekg(0, engine_file.beg);

	std::vector<char> engine_data(fsize);
	engine_file.read(engine_data.data(), fsize);
	if (!engine_file)
	{
		std::cout << "Error loading engine file: " << plan_file << std::endl;
		return nullptr;
	}

	auto runtime = std::unique_ptr < nvinfer1::IRuntime, std::function<void(nvinfer1::IRuntime*)> >(
		nvinfer1::createInferRuntime(logger),
		[](nvinfer1::IRuntime* t) {
			t->destroy();
		}
	);

	if (!runtime)
	{
		std::cout << "create Infer Runtime failed !" << std::endl;
		return nullptr;
	}

	nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), fsize, plugin_factory);
	if (!engine)
	{
		std::cout << "create Cuda Engine failed !" << std::endl;
		return nullptr;
	}

	std::cout << "Loading engine file Complete!" << std::endl;

	return engine;
}

std::string dims_to_string(const nvinfer1::Dims d)
{
	std::stringstream s;
	assert(d.nbDims >= 1);
	for (int i = 0; i < d.nbDims - 1; ++i)
	{
		s << std::setw(4) << d.d[i] << " x";
	}
	s << std::setw(4) << d.d[d.nbDims - 1];

	return s.str();
}

void print_layer_info(int layer_idx, std::string layer_name, nvinfer1::Dims input_dims,
	nvinfer1::Dims output_dims, size_t weight_ptr)
{
	std::cout << std::setw(6) << std::left << layer_idx << std::setw(15) << std::left << layer_name;
	std::cout << std::setw(20) << std::left << dims_to_string(input_dims) << std::setw(20) << std::left
		<< dims_to_string(output_dims);
	std::cout << std::setw(6) << std::left << weight_ptr << std::endl;
}

void print_layer_info(std::string layer_idx, std::string layer_name, std::string input_dims, std::string output_dims, std::string weight_ptr)
{
	std::cout << std::setw(6) << std::left << layer_idx << std::setw(15) << std::left << layer_name;
	std::cout << std::setw(20) << std::left << input_dims << std::setw(20) << std::left
		<< output_dims;
	std::cout << std::setw(6) << std::left << weight_ptr << std::endl;
}
