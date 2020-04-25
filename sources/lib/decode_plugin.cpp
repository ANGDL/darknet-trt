#include "decode_plugin.h"
#include "decode_kernel.cuh"

void darknet::DecodePlugin::deserialize(void const* data, size_t length)
{
	const char* d = static_cast<const char*>(data);
	read(d, score_thresh);
	read(d, top_n);
	size_t anchors_size;
	read(d, anchors_size);
	while (anchors_size--)
	{
		float val;
		read(d, val);
		anchors.push_back(val);
	}

	read(d, stride);

	read(d, grid_size);
	read(d, num_anchors);
	read(d, num_classes);
}

void darknet::DecodePlugin::serialize(void* buffer)
{
	char* d = static_cast<char*> (buffer);
	write(d, score_thresh);
	write(d, top_n);
	write(d, anchors.size());
	for (auto& val : anchors) {
		write(d, val);
	}

	write(d, stride);
	write(d, grid_size);
	write(d, num_anchors);
	write(d, num_classes);
}

size_t darknet::DecodePlugin::getSerializationSize()
{
	return sizeof(score_thresh) + sizeof(top_n) + sizeof(size_t) +
		sizeof(float) * anchors.size() + sizeof(stride) + sizeof(grid_size) +
		sizeof(num_anchors) + sizeof(num_classes);
}

int darknet::DecodePlugin::initialize()
{
	return 0;
}

int darknet::DecodePlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
	return cuda_decode_layer(
		inputs[0],
		outputs,
		batchSize,
		stride,
		grid_size,
		num_anchors,
		num_classes,
		anchors,
		score_thresh,
		top_n,
		workspace,
		this->getWorkspaceSize(batchSize),
		stream
	);
}

void darknet::DecodePlugin::configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize)
{
	assert(nbInputs == 1);
	assert(nbOutputs == 3);
	assert(inputDims != nullptr && inputDims[0].nbDims == 3);
	assert(num_anchors * (5 + num_classes) == inputDims[0].d[0]);
	assert(grid_size == inputDims[0].d[1]);
	assert(grid_size == inputDims[0].d[2]);
}

darknet::DecodePlugin::DecodePlugin(float score_thresh, int top_n, std::vector<float>const& anchors,
	int stride, size_t grid_size, size_t num_anchors, size_t num_classes) :
	score_thresh(score_thresh), top_n(top_n), anchors(anchors),
	stride(stride), grid_size(grid_size), num_anchors(num_anchors), num_classes(num_classes)
{

}

darknet::DecodePlugin::DecodePlugin(void const* data, size_t length)
{
	this->deserialize(data, length);
}

void darknet::DecodePlugin::terminate()
{
}

int darknet::DecodePlugin::getNbOutputs() const
{
	return 3;
}

nvinfer1::Dims darknet::DecodePlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
	assert(nbInputDims == 1);
	assert(index < this->getNbOutputs());
	return Dims3(top_n * (index == 1 ? 4 : 1), 1, 1);
}

size_t darknet::DecodePlugin::getWorkspaceSize(int maxBatchSize) const
{
	return cuda_decode_layer(
		nullptr,
		nullptr,
		maxBatchSize,
		stride,
		grid_size,
		num_anchors,
		num_classes,
		anchors,
		score_thresh,
		top_n,
		nullptr,
		0,
		nullptr
	);
}
