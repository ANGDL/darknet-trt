#include "upsample_plugin.h"
#include "upsample_kernel.cuh"

darknet::UpsampleLayer::UpsampleLayer(float stride, const nvinfer1::Dims input_dim) :
	stride(stride),
	in_c(input_dim.d[0]),
	in_h(input_dim.d[1]),
	in_w(input_dim.d[2])
{

}

darknet::UpsampleLayer::UpsampleLayer(const void* data, size_t len)
{
	assert(data != nullptr && len > 0);
	const char* p = reinterpret_cast<const char*>(data), * a = p;
	read(p, stride);
	read(p, in_c);
	read(p, in_h);
	read(p, in_w);
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
	return DimsCHW{ c, h ,w };
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
	NV_CUDA_CHECK(cuda_upsample_layer(inputs[0], outputs[0], batchSize, stride, in_c, in_h, in_w, stream));
	return 0;
}

size_t darknet::UpsampleLayer::getSerializationSize()
{
	return sizeof(stride) + sizeof(in_c) + sizeof(in_h) + sizeof(in_w);
}

void darknet::UpsampleLayer::serialize(void* buffer)
{
	assert(buffer != nullptr);
	char* p = reinterpret_cast<char*>(buffer), * a = p;
	write(p, stride);
	write(p, in_c);
	write(p, in_h);
	write(p, in_w);
	assert(p == a + getSerializationSize());
}

void darknet::UpsampleLayer::configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize)
{
	assert(nbInputs == 1);
	assert(inputDims != nullptr && inputDims[0].nbDims == 3);
	assert(in_c == inputDims[0].d[0]);
	assert(in_h == inputDims[0].d[1]);
	assert(in_w == inputDims[0].d[2]);
}
