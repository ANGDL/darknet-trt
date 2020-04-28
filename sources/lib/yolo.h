#ifndef _YOLO_H_
#define  _YOLO_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <set>
#include <memory>
#include "darknet_cfg.h"
#include "NvInfer.h"
#include "plugin_factory.h"


namespace darknet {
	class Logger : public nvinfer1::ILogger
	{
	public:
		void log(nvinfer1::ILogger::Severity severity, const char* msg) override
		{
			// suppress info-level messages
			if (severity == Severity::kINFO) return;

			switch (severity)
			{
			case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
			case Severity::kERROR: std::cerr << "ERROR: "; break;
			case Severity::kWARNING: std::cerr << "WARNING: "; break;
			case Severity::kINFO: std::cerr << "INFO: "; break;
			default: std::cerr << "UNKNOWN: "; break;
			}
			std::cerr << msg << std::endl;
		}
	};

	struct InferDeleter
	{
		template <typename T>
		void operator()(T* obj) const
		{
			if (obj)
			{
				obj->destroy();
			}
		}
	};

	template <typename T>
	using unique_ptr_infer = std::unique_ptr<T, InferDeleter>;

	class YoloTinyMaxpoolPaddingFormula : public nvinfer1::IOutputDimensionsFormula
	{
	public:
		void add_same_padding_layer(std::string input) {
			same_pooling_layers.insert(input);
		}

	private:

		nvinfer1::DimsHW compute(DimsHW input_dims, DimsHW kernel_size, DimsHW stride, DimsHW padding, DimsHW dilation, const char* layerName) const {
			int output_dim;
			// same padding
			if (same_pooling_layers.find(layerName) != same_pooling_layers.end()) {
				output_dim = (input_dims.d[0] + 2 * padding.d[0]) / stride.d[0];
			}
			//valid padding
			else {
				output_dim = (input_dims.d[0] - kernel_size.d[0]) / stride.d[0] + 1;
			}

			return nvinfer1::DimsHW(output_dim, output_dim);
		}

		std::set<std::string> same_pooling_layers;
	};

	class Yolo {
	public:
		Yolo(NetConfig* config, uint batch_size, float confidence_thresh, float nms_thresh);
		~Yolo();
		bool good() const;

	protected:
		std::unique_ptr<NetConfig> config;
		uint batch_size;
		float prob_thresh;
		float nms_thresh;

		int input_index;

		nvinfer1::ICudaEngine* engine;
		nvinfer1::IExecutionContext* context;
		cudaStream_t cuda_stream;
		PluginFactory* plugin_factory;
		std::vector<void*> bindings;
		std::vector<float*> trt_output_buffers;
		std::unique_ptr<YoloTinyMaxpoolPaddingFormula> tiny_maxpool_padding_formula;

		Logger logger;

		bool is_init;

	private:
		bool build(const nvinfer1::DataType data_type, const std::string planfile_path /*, Int8EntropyCalibrator* calibrator*/);

		nvinfer1::ILayer* add_maxpool(int layer_idx, const darknet::Block& block, nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network);

		nvinfer1::ILayer* add_conv_bn_leaky(
			int layer_idx,
			const darknet::Block& block,
			std::vector<float>& weights,
			std::vector<nvinfer1::Weights>& trt_weights,
			int& weight_ptr,
			int& input_channels,
			nvinfer1::ITensor* input,
			nvinfer1::INetworkDefinition* network
		);

		nvinfer1::ILayer* add_conv_linear(
			int layer_idx,
			const darknet::Block& block,
			std::vector<float>& weights,
			std::vector<nvinfer1::Weights>& trt_weights,
			int& weight_ptr,
			int& input_channels,
			nvinfer1::ITensor* input,
			nvinfer1::INetworkDefinition* network
		);

		nvinfer1::ILayer* add_upsample(
			int layer_idx,
			const darknet::Block& block,
			std::vector<float>& weights,
			std::vector<nvinfer1::Weights>& trt_weights,
			int& weight_ptr,
			int& input_channels,
			nvinfer1::ITensor* input,
			nvinfer1::INetworkDefinition* network
		);

		nvinfer1::IConvolutionLayer* add_conv(
			int layer_idx,
			int filters,
			int kernel_size,
			int stride,
			int pad,
			std::vector<float>& weight,
			int& weight_ptr,
			int& input_channels,
			nvinfer1::ITensor* input,
			nvinfer1::INetworkDefinition* network,
			bool use_biases
		);

		nvinfer1::IScaleLayer* add_bn(
			int layer_idx,
			int filters,
			std::vector<float>& bn_biases,
			std::vector<float>& bn_weights,
			std::vector<float>& bn_mean,
			std::vector<float>& bn_var,
			nvinfer1::ITensor* input,
			nvinfer1::INetworkDefinition* network
		);

		nvinfer1::IPluginLayer* add_leakyRelu(
			int layer_idx,
			nvinfer1::ITensor* input,
			nvinfer1::INetworkDefinition* network
		);

		nvinfer1::IPluginV2Layer* add_decode(
			nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network, std::string name,
			float score_thresh,
			const std::vector<float> anchors,
			int stride,
			int gride_size,
			int num_anchors,
			int num_classes
		);


		nvinfer1::ILayer* add_upsample2(
			int layer_idx,
			const darknet::Block& block,
			std::vector<float>& weights,
			int& input_channels,
			nvinfer1::ITensor* input,
			nvinfer1::INetworkDefinition* network
		);

		nvinfer1::ILayer* netAddConvBNLeaky(int layerIdx, const darknet::Block& block,
			std::vector<float>& weights,
			std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr,
			int& inputChannels, nvinfer1::ITensor* input,
			nvinfer1::INetworkDefinition* network);

		nvinfer1::ILayer* netAddConvLinear(int layerIdx, const darknet::Block& block,
			std::vector<float>& weights,
			std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr,
			int& inputChannels, nvinfer1::ITensor* input,
			nvinfer1::INetworkDefinition* network);
	};

}

#endif
