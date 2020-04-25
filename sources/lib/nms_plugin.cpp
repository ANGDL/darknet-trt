#include "nms_plugin.h"
#include "nms_kernel.cuh"

void darknet::NmsPlugin::deserialize(void const* data, size_t length)
{
	const char* d = static_cast<const char*>(data);
	read(d, nms_thresh);
	read(d, detections_per_im);
	read(d, count);
}

void darknet::NmsPlugin::serialize(void* buffer)
{
	char* d = static_cast<char*>(buffer);
	write(d, nms_thresh);
	write(d, detections_per_im);
	write(d, count);
}

size_t darknet::NmsPlugin::getSerializationSize()
{
	return sizeof(nms_thresh) + sizeof(detections_per_im) + sizeof(count);
}

int darknet::NmsPlugin::initialize()
{
	return 0;
}

int darknet::NmsPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
	return cuda_nms(
		batchSize,
		inputs,
		outputs,
		count,
		detections_per_im,
		nms_thresh,
		workspace,
		this->getWorkspaceSize(batchSize),
		stream
	);
}

void darknet::NmsPlugin::configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize)
{
	assert(inputDims != nullptr && nbInputs == 3);
	assert(inputDims[0].d[0] == inputDims[2].d[0]);
	assert(inputDims[0].d[0] == inputDims[1].d[0] * 4);
	count = inputDims[0].d[0];
}

darknet::NmsPlugin::NmsPlugin(void const* data, size_t length)
{
	this->deserialize(data, length);
}

darknet::NmsPlugin::NmsPlugin(float nms_thresh, int dections_per_im):
	nms_thresh(nms_thresh), detections_per_im(dections_per_im), count(0)
{
}

darknet::NmsPlugin::NmsPlugin(float nms_thresh, int dections_per_im, size_t count):
	nms_thresh(nms_thresh), detections_per_im(dections_per_im), count(count)
{
}

void darknet::NmsPlugin::terminate()
{
}

int darknet::NmsPlugin::getNbOutputs() const
{
	return 3;
}

nvinfer1::Dims darknet::NmsPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
	assert(nbInputDims == 3);
	assert(index < this->getNbOutputs());
	return Dims3(detections_per_im * (index == 1 ? 4 : 1), 1, 1);
}

size_t darknet::NmsPlugin::getWorkspaceSize(int maxBatchSize) const
{
	static int size = -1;
	if (size == -1)
	{
		size = cuda_nms(maxBatchSize, nullptr, nullptr, count, detections_per_im, nms_thresh, nullptr, 0, nullptr);
	}
	return size;
}
