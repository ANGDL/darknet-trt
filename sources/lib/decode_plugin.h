#ifndef _DECODE_PLUGIN_H_
#define _DECODE_PLUGIN_H_

#include "NvInferPlugin.h"
#include "darknet_utils.h"

namespace darknet {
	using namespace nvinfer1;

	class DecodePlugin : public nvinfer1::IPlugin
	{
	private:
		float score_thresh;
		int top_n;
		std::vector<float> anchors;

		float stride;

		size_t grid_size;
		size_t num_anchors;
		size_t num_classes;

	protected:
		void deserialize(void const* data, size_t length);
		void serialize(void* buffer) override;
		size_t getSerializationSize() override;

		int initialize() override;
		int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;
		void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override;

	public:
		DecodePlugin(float score_thresh, int top_n, std::vector<float>const& anchors, int stride,
			size_t grid_size, size_t num_anchors, size_t num_classes
		);

		DecodePlugin(void const* data, size_t length);

		void terminate() override;
		int getNbOutputs() const override;
		Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
		size_t getWorkspaceSize(int maxBatchSize) const override;
	};
}

#endif
