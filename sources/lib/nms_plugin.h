#ifndef _NMS_PLUGIN_H_
#define _NMS_PLUGIN_H_

#include "NvInferPlugin.h"
#include "darknet_utils.h"
#include "nms_kernel.cuh"

namespace darknet {
	using namespace nvinfer1;

	class NMSPlugin : public IPluginV2Ext {
	private:
		float nms_thresh;
		int detections_per_im;

		size_t count;

	protected:
		void deserialize(void const* data, size_t length) {
			const char* d = static_cast<const char*>(data);
			read(d, nms_thresh);
			read(d, detections_per_im);
			read(d, count);
		}
		void serialize(void* buffer) const override {
			char* d = static_cast<char*>(buffer);
			write(d, nms_thresh);
			write(d, detections_per_im);
			write(d, count);
		}
		size_t getSerializationSize() const override {
			return sizeof(nms_thresh) + sizeof(detections_per_im) + sizeof(count);
		}

	public:
		NMSPlugin(void const* data, size_t length) {
			this->deserialize(data, length);
		}

		NMSPlugin(float nms_thresh, int dections_per_im) :
			nms_thresh(nms_thresh), detections_per_im(dections_per_im), count(0)
		{}

		NMSPlugin(float nms_thresh, int dections_per_im, size_t count) :
			nms_thresh(nms_thresh), detections_per_im(dections_per_im), count(count)
		{}

		const char* getPluginType() const override {
			return "NMS";
		}

		const char* getPluginVersion() const override {
			return "1";
		}

		int getNbOutputs() const override {
			return 3;
		}

		Dims getOutputDimensions(int index,
			const Dims* inputs, int nbInputDims) override {
			assert(nbInputDims == 3);
			assert(index < this->getNbOutputs());
			return Dims3(detections_per_im * (index == 1 ? 4 : 1), 1, 1);
		}

		bool supportsFormat(DataType type, PluginFormat format) const override {
			return type == DataType::kFLOAT && format == PluginFormat::kLINEAR;
		}

		int initialize() override { return 0; }

		void terminate() override {}

		size_t getWorkspaceSize(int maxBatchSize) const override {
			return cuda_nms(maxBatchSize, nullptr, nullptr, count, detections_per_im, nms_thresh, nullptr, 0, nullptr);
		}

		int enqueue(int batchSize,
			const void* const* inputs, void** outputs,
			void* workspace, cudaStream_t stream) override {

			//size_t pred_size = count ;
			//float* test_input;
			//cudaMallocHost(&test_input, pred_size * sizeof(float));
			//cudaMemcpy(test_input, inputs[0], pred_size * sizeof(float), cudaMemcpyDeviceToHost);

			//for (int i = 0; i < pred_size; ++i) {
			//	printf("%f ", test_input[i]);
			//}
			//printf("\n ");
			//cudaFreeHost(test_input);

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

		void destroy() override {
			delete this;
		}

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
			assert(nbInputs == 3);
			assert(inputDims[0].d[0] == inputDims[2].d[0]);
			assert(inputDims[1].d[0] == inputDims[2].d[0] * 4);
			count = inputDims[0].d[0];
		}

		IPluginV2Ext* clone() const override {
			return new NMSPlugin(nms_thresh, detections_per_im, count);
		}

	};

	class NMSPluginCreator : public IPluginCreator {
	public:
		NMSPluginCreator() {}

		const char* getPluginNamespace() const override {
			return "";
		}
		const char* getPluginName() const override {
			return "NMS";
		}

		const char* getPluginVersion() const override {
			return "1";
		}

		IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override {
			return new NMSPlugin(serialData, serialLength);
		}

		void setPluginNamespace(const char* N) override {}
		const PluginFieldCollection* getFieldNames() override { return nullptr; }
		IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override { return nullptr; }
	};

	REGISTER_TENSORRT_PLUGIN(NMSPluginCreator);
}

#endif
