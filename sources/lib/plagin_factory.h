#ifndef _PLAGIN_FACTORY_H_
#define _PLAGIN_FACTORY_H_

#include "NvInferPlugin.h"
#include "darknet_utils.h"
#include "yolo_layer_kernels.cuh"

namespace darknet {
	using namespace nvinfer1;
	class YoloLayer : public IPlugin
	{
	public:
		YoloLayer(
			unsigned int num_bboxes,
			unsigned int num_classes,
			unsigned int gride_size
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
		template<typename T>
		void write(char*& buffer, const T& val) {
			*reinterpret_cast<T*>(buffer) = val;
			buffer += sizeof(T);
		}

		template<typename T>
		void read(const char*& buffer, T& val) {
			val = *reinterpret_cast<const T*>(buffer);
			buffer += sizeof(T);
		}

		unsigned int num_bboxes;
		unsigned int num_classes;
		unsigned int grid_size;
		size_t output_size;
	};

}


#endif
