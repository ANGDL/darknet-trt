#ifndef _NMS_PLUGIN_H_
#define _NMS_PLUGIN_H_

#include "NvInferPlugin.h"
#include "darknet_utils.h"

namespace darknet {
	using namespace nvinfer1;

	class NmsPlugin : public IPlugin {
	private:
		float nms_thresh;
		int detections_per_im;

		size_t count;

	protected:
		void deserialize(void const* data, size_t length);
		void serialize(void* buffer) override;
		size_t getSerializationSize() override;

		int initialize() override;
		int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;
		void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override;

	public:
		NmsPlugin(void const* data, size_t length);
		NmsPlugin(float nms_thresh, int dections_per_im);
		NmsPlugin(float nms_thresh, int dections_per_im, size_t count);

		void terminate() override;
		int getNbOutputs() const override;
		Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
		size_t getWorkspaceSize(int maxBatchSize) const override;
	};
}

#endif

