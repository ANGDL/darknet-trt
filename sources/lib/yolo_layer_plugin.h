#ifndef _YOLO_PLUGIN_H_
#define _YOLO_PLUGIN_H_
#include "NvInferPlugin.h"
#include "darknet_utils.h"

namespace darknet {
	using namespace nvinfer1;

	// YOLO Layer
	class YoloLayer : public IPlugin
	{
	public:
		YoloLayer(
			unsigned int num_bboxes,
			unsigned int num_classes,
			unsigned int grid_size
		);
		YoloLayer(const void* data, size_t len);

		int getNbOutputs() const override;
		Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
		int initialize() override;
		void terminate() override;
		size_t getWorkspaceSize(int maxBatchSize) const override;
		int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;
		size_t getSerializationSize() override;
		void serialize(void* buffer) override;
		void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override;

	private:


		unsigned int num_bboxes;
		unsigned int num_classes;
		unsigned int grid_size;
		size_t output_size;
	};
}

#endif
