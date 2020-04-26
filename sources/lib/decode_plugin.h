#ifndef _DECODE_PLUGIN_H_
#define _DECODE_PLUGIN_H_

#include "NvInferPlugin.h"
#include "darknet_utils.h"
#include "decode_kernel.cuh"

namespace darknet {
	using namespace nvinfer1;

	class DecodePlugin : public nvinfer1::IPluginV2Ext
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
		void deserialize(void const* data, size_t length) {
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
		void serialize(void* buffer) const  override {
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
		size_t getSerializationSize() const  override {
			return sizeof(score_thresh) + sizeof(top_n) + sizeof(size_t) +
				sizeof(float) * anchors.size() + sizeof(stride) + sizeof(grid_size) +
				sizeof(num_anchors) + sizeof(num_classes);
		}

	public:
		darknet::DecodePlugin::DecodePlugin(float score_thresh, int top_n, std::vector<float>const& anchors,
			int stride, size_t grid_size, size_t num_anchors, size_t num_classes) :
			score_thresh(score_thresh), top_n(top_n), anchors(anchors),
			stride(stride), grid_size(grid_size), num_anchors(num_anchors), num_classes(num_classes) {

		}

		darknet::DecodePlugin::DecodePlugin(void const* data, size_t length) {
			this->deserialize(data, length);
		}

		const char* getPluginType() const override {
			return "YoloDecode";
		}

		const char* getPluginVersion() const override {
			return "1";
		}

		int getNbOutputs() const override {
			return 3;
		}

		Dims getOutputDimensions(int index,
			const Dims* inputs, int nbInputDims) override {
			assert(nbInputDims == 1);
			assert(index < this->getNbOutputs());
			return Dims3(top_n * (index == 1 ? 4 : 1), 1, 1);
		}

		bool supportsFormat(DataType type, PluginFormat format) const override {
			return type == DataType::kFLOAT && format == PluginFormat::kLINEAR;
		}

		int initialize() override { return 0; }

		void terminate() override {}

		size_t getWorkspaceSize(int maxBatchSize) const override {
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

		int enqueue(int batchSize,
			const void* const* inputs, void** outputs,
			void* workspace, cudaStream_t stream) override {
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
			assert(*inputTypes == nvinfer1::DataType::kFLOAT &&
				floatFormat == nvinfer1::PluginFormat::kLINEAR);
			assert(nbInputs == 1);
			assert(nbOutputs == 3);
			assert(inputDims != nullptr && inputDims[0].nbDims == 3);
			assert(num_anchors * (5 + num_classes) == inputDims[0].d[0]);
			assert(grid_size == inputDims[0].d[1]);
			assert(grid_size == inputDims[0].d[2]);
		}

		IPluginV2Ext* clone() const override {
			return new DecodePlugin(score_thresh, top_n, anchors, stride, grid_size, num_anchors, num_classes);
		}
	};

	class DecodePluginCreator : public IPluginCreator {
	public:
		DecodePluginCreator() {}

		const char* getPluginName() const override {
			return "YoloDecode";
		}

		const char* getPluginVersion() const override {
			return "1";
		}

		const char* getPluginNamespace() const override {
			return "";
		}

		IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override {
			return new DecodePlugin(serialData, serialLength);
		}

		void setPluginNamespace(const char* N) override {}
		const PluginFieldCollection* getFieldNames() override { return nullptr; }
		IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override { return nullptr; }
	};

	REGISTER_TENSORRT_PLUGIN(DecodePluginCreator);
}

#endif
