#include "yolo_layer_plugin.h"
#include "yolo_layer_kernels.cuh"


// ͨ��API���û�������ṹ������, 
// ���������Զ��壬��pluginfactory���������ù���
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

// ͨ�������л���ɲ���������е�plugin����
// ����ͨ����ȡ���л�����ļ����з����л��Ĺ���
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

// �������
int darknet::YoloLayer::getNbOutputs() const
{
	return 1;
}

// �����ά��
// ����yolo��ֻ�Ƕ�confidence �� local ���� sigmod �� exp�Ĵ���
// ���ԣ�����������ά����һ����
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

// ִ�иò�
int darknet::YoloLayer::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{

	NV_CUDA_CHECK(cuda_yolo_layer(
		inputs[0], outputs[0], batchSize, grid_size, num_classes, num_bboxes, output_size, stream));

	return 0;
}

// ��øò��������ʱ�Դ��С��
size_t darknet::YoloLayer::getSerializationSize()
{
	return sizeof(num_bboxes) + sizeof(num_classes) + sizeof(grid_size) + sizeof(output_size);
}

// �Ըò���г�ʼ������engine����ʱ�����á�
void darknet::YoloLayer::serialize(void* buffer)
{
	char* p = reinterpret_cast<char*>(buffer), * a = p;
	write(p, num_bboxes);
	write(p, num_classes);
	write(p, grid_size);
	write(p, output_size);
	assert(p == a + getSerializationSize());
}

// ���øò�Ĳ������ú�����initialize()����֮ǰ�����������á�
// ��Ϊ�ò��ṩ��һ�����ᣬ���Ը�����Ȩ�ء��ߴ�����������С�������㷨ѡ��
void darknet::YoloLayer::configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize)
{
	assert(nbInputs == 1);
	assert(inputDims != nullptr && inputDims[0].nbDims == 3);
}