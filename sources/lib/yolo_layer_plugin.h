#ifndef _YOLO_PLUGIN_H_
#define _YOLO_PLUGIN_H_

#include "NvInferPlugin.h"
#include "darknet_utils.h"
#include "yolo_layer_kernels.cuh"

namespace darknet {
	using namespace nvinfer1;

	class YoloLayerPlugin : public nvinfer1::IPluginV2Ext
	{
	private:
		unsigned int num_bboxes;
		unsigned int num_classes;
		unsigned int grid_size;
		size_t output_size;

	protected:
		void deserialize(void const* data, size_t length) {
			const char* d = reinterpret_cast<const char*>(data), * a = d;
			read(d, num_bboxes);
			read(d, num_classes);
			read(d, grid_size);
			read(d, output_size);
		}
		void serialize(void* buffer) const  override {
			char* p = reinterpret_cast<char*>(buffer), * a = p;
			write(p, num_bboxes);
			write(p, num_classes);
			write(p, grid_size);
			write(p, output_size);
		}
		size_t getSerializationSize() const  override {
			return sizeof(num_bboxes) + sizeof(num_classes) + sizeof(grid_size) + sizeof(output_size);
		}

	public:
		YoloLayerPlugin(unsigned int num_bboxes,
			unsigned int num_classes,
			unsigned int grid_size) :

			num_bboxes(num_bboxes),
			num_classes(num_classes),
			grid_size(grid_size),
			output_size((size_t)grid_size* grid_size* num_bboxes* (5 + (size_t)num_classes))
		{

		}

		YoloLayerPlugin(void const* data, size_t length) {
			this->deserialize(data, length);
		}

		const char* getPluginType() const override {
			return "YoloLayer";
		}

		const char* getPluginVersion() const override {
			return "1";
		}

		int getNbOutputs() const override {
			return 1;
		}

		Dims getOutputDimensions(int index,
			const Dims* inputs, int nbInputDims) override {
			assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
			return inputs[0];
		}

		bool supportsFormat(DataType type, PluginFormat format) const override {
			return type == DataType::kFLOAT && format == PluginFormat::kLINEAR;
		}

		int initialize() override { return 0; }

		void terminate() override {}

		size_t getWorkspaceSize(int maxBatchSize) const override {
			return 0;
		}

		int enqueue(int batchSize,
			const void* const* inputs, void** outputs,
			void* workspace, cudaStream_t stream) override {
			NV_CUDA_CHECK(cuda_yolo_layer(
				inputs[0], outputs[0], batchSize, grid_size, num_classes, num_bboxes, output_size, stream));

			return 0;
		}

		void destroy() override {
			delete this;
		};

		const char* getPluginNamespace() const override {
			return "";
		}

		void setPluginNamespace(const char* N) override {

		}

		// IPluginV2Ext Methods
		DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const
		{
			assert(index < 3);
			return DataType::kFLOAT;
		}

		bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted,
			int nbInputs) const {
			return false;
		}

		bool canBroadcastInputAcrossBatch(int inputIndex) const { return false; }

		void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
			const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
			const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
		{
			assert(nbInputs == 1);
			assert(inputDims != nullptr && inputDims[0].nbDims == 3);
		}

		IPluginV2Ext* clone() const override {
			return new YoloLayerPlugin(num_bboxes, num_classes, grid_size);
		}
	};

	class YoloLayerPluginCreator : public IPluginCreator {
	public:
		YoloLayerPluginCreator() {}

		const char* getPluginName() const override {
			return "YoloLayer";
		}

		const char* getPluginVersion() const override {
			return "1";
		}

		const char* getPluginNamespace() const override {
			return "";
		}

		IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override {
			return new YoloLayerPlugin(serialData, serialLength);
		}

		void setPluginNamespace(const char* N) override {}
		const PluginFieldCollection* getFieldNames() override { return nullptr; }
		IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override { return nullptr; }
	};

	REGISTER_TENSORRT_PLUGIN(YoloLayerPluginCreator);

}

#endif
