#ifndef _UPSAMPLE_PLUGIN_H_
#define _UPSAMPLE_PLUGIN_H_

#include "NvInferPlugin.h"
#include "darknet_utils.h"

namespace darknet {
	using namespace nvinfer1;

	class UpsampleLayer : public IPlugin
	{
	public:
		UpsampleLayer(float stride, const nvinfer1::Dims input_dim);
		UpsampleLayer(const void* data, size_t len);

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
		float stride;
		int in_c;
		int in_h;
		int in_w;
	};

}


#endif // 
