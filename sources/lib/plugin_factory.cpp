#include <assert.h>

#include "plugin_factory.h"

// 通过API调用或者网络结构构造类, 
// 函数参数自定义，在pluginfactory中自主调用构造
darknet::YoloLayer::YoloLayer(
	unsigned int num_bboxes,
	unsigned int num_classes,
	unsigned int grid_size) :

	num_bboxes(num_bboxes),
	num_classes(num_classes),
	grid_size(grid_size),
	output_size((size_t)grid_size* grid_size* num_bboxes* (5 + (size_t)num_classes))
{

}

// 通过反序列化完成参数读入进行的plugin构造
// 就是通过读取序列化后的文件进行反序列化的构造
darknet::YoloLayer::YoloLayer(const void* data, size_t len)
{
	assert(data != nullptr && len > 0);
	const char* d = reinterpret_cast<const char*>(data), * a = d;
	read(d, num_bboxes);
	read(d, num_classes);
	read(d, grid_size);
	read(d, output_size);
	assert(d == a + len);
}

// 几个输出
int darknet::YoloLayer::getNbOutputs() const
{
	return 1;
}

// 输出的维度
// 这里yolo层只是对confidence 和 local 做了 sigmod 和 exp的处理
// 所以，输入和输出的维度是一样的
nvinfer1::Dims darknet::YoloLayer::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims)
{
	assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
	return inputs[0];
}

int darknet::YoloLayer::initialize()
{
	return 0;
}

void darknet::YoloLayer::terminate()
{
}

size_t darknet::YoloLayer::getWorkspaceSize(int maxBatchSize) const
{
	return 0;
}

// 执行该层
int darknet::YoloLayer::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{

	NV_CUDA_CHECK(cuda_yolo_layer(
		inputs[0], outputs[0], batchSize, grid_size, num_classes, num_bboxes, output_size, stream));

	return 0;
}

// 获得该层所需的临时显存大小。
size_t darknet::YoloLayer::getSerializationSize()
{
	return sizeof(num_bboxes) * +sizeof(num_classes) + sizeof(grid_size) + sizeof(output_size);
}

// 对该层进行初始化，在engine创建时被调用。
void darknet::YoloLayer::serialize(void* buffer)
{
	char* p = reinterpret_cast<char*>(buffer), * a = p;
	write(p, num_bboxes);
	write(p, num_classes);
	write(p, grid_size);
	write(p, output_size);
	assert(p == a + getSerializationSize());
}

// 配置该层的参数。该函数在initialize()函数之前被构造器调用。
// 它为该层提供了一个机会，可以根据其权重、尺寸和最大批量大小来做出算法选择。
void darknet::YoloLayer::configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize)
{
	assert(nbInputs == 1);
	assert(inputDims != nullptr && inputDims[0].nbDims == 3);
}


darknet::PluginFactory::PluginFactory()
{

}

nvinfer1::IPlugin* darknet::PluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength)
{
	if (std::string(layerName).find("leaky") != std::string::npos) {
		unique_ptr_nvplugin leaky = unique_ptr_nvplugin(nvinfer1::plugin::createPReLUPlugin(serialData, serialLength));
		leakyReLU_layers.push_back(leaky);
		return leaky.get();
	}
	else if (std::string(layerName).find("yolo") != std::string::npos) {
		unique_ptr_iplugin yolo = unique_ptr_iplugin(new YoloLayer(serialData, serialLength));
		yolo_layers.push_back(yolo);
		return yolo.get();
	}
	else if (std::string(layerName).find("upsample") != std::string::npos) {
		unique_ptr_iplugin upsample = unique_ptr_iplugin(new UpsampleLayer(serialData, serialLength));
		upsample_layers.push_back(upsample);
		return upsample.get();
	}
	else {
		std::cerr << "ERROR: Unrecognised layer : " << layerName << std::endl;
		assert(0);
		return nullptr;
	}
}


darknet::UpsampleLayer::UpsampleLayer(float stride, const nvinfer1::Dims input_dim) :
	stride(stride),
	in_dims(input_dim)
{

}

darknet::UpsampleLayer::UpsampleLayer(const void* data, size_t len)
{
	assert(data != nullptr && len > 0);
	const char* p = reinterpret_cast<const char*>(data), * a = p;
	read(p, stride);
	read(p, in_dims);
	assert(p == a + len);
}

int darknet::UpsampleLayer::getNbOutputs() const
{
	return 1;
}

nvinfer1::Dims darknet::UpsampleLayer::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
	assert(index == 0 && inputs != nullptr && nbInputDims == 1 && inputs[0].nbDims == 3);
	int c = inputs[0].d[0];
	int h = inputs[0].d[1] * stride;
	int w = inputs[0].d[2] * stride;
	return Dims{ c, h ,w };
}

int darknet::UpsampleLayer::initialize()
{
	return 0;
}

void darknet::UpsampleLayer::terminate()
{

}

size_t darknet::UpsampleLayer::getWorkspaceSize(int maxBatchSize) const
{
	return 0;
}

int darknet::UpsampleLayer::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
	NV_CUDA_CHECK(cuda_upsample_layer(inputs[0], outputs[0], batchSize, stride, in_dims, stream));
	return 0;
}

size_t darknet::UpsampleLayer::getSerializationSize()
{
	return sizeof(stride);
}

void darknet::UpsampleLayer::serialize(void* buffer)
{
	assert(buffer != nullptr);
	char* p = reinterpret_cast<char*>(buffer), * a = p;
	write(p, stride);
	write(p, in_dims);
	assert(p == a + getSerializationSize());
}

void darknet::UpsampleLayer::configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize)
{
	assert(nbInputs == 1);
	assert(inputDims != nullptr && inputDims[0].nbDims == 3);
	assert(in_dims.d[0] == inputDims[0].d[0]);
	assert(in_dims.d[1] == inputDims[0].d[1]);
	assert(in_dims.d[2] == inputDims[0].d[2]);
}
