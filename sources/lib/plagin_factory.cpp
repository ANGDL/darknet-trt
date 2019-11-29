#include <assert.h>

#include "plagin_factory.h"

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
	const char* d = reinterpret_cast<const char*>(data);
	read(d, num_bboxes);
	read(d, num_classes);
	read(d, grid_size);
	read(d, output_size);
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
