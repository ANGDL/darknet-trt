#ifndef _YOLO_H_
#define  _YOLO_H_

#include <set>
#include <memory>
#include "darknet_cfg.h"
#include "NvInfer.h"
#include "plugin_factory.h"

namespace darknet {

	class YoloTinyMaxpoolPaddingFormula : public nvinfer1::IOutputDimensionsFormula
	{
	public:
		void add_same_padding_layer(std::string input) {
			same_pooling_layers.insert(input);
		}

	private:

		nvinfer1::DimsHW compute(DimsHW inputDims, DimsHW kernelSize, DimsHW stride, DimsHW padding, DimsHW dilation, const char* layerName) const TRTNOEXCEPT = 0 {
			int output_dim;
			if (same_pooling_layers.find(layerName) != same_pooling_layers.end()) {
				output_dim = (inputDims.d[0] + 2 * padding.d[0] - kernelSize.d[0]) / stride.d[0];
			}
			else {
				output_dim = (inputDims.d[0] - kernelSize.d[0]) / stride.d[0];
			}

			return nvinfer1::DimsHW(output_dim, output_dim);
		}

		std::set<std::string> same_pooling_layers;
	};

	class Yolo {
	public:
		Yolo(NetConfig* config, float confidence_thresh, float nms_thresh);

	protected:
		std::unique_ptr<NetConfig> config;
		float prob_thresh;
		float nms_thresh;

		int input_index;

		nvinfer1::ICudaEngine* engine;
		nvinfer1::IExecutionContext* context;
		cudaStream_t cuda_stream;
		PluginFactory* plugin_factory;
		std::unique_ptr<IOutputDimensionsFormula> tiny_maxpool_padding_formula;


	private:
		void create_yolo_engine(const nvinfer1::DataType data_type = nvinfer1::DataType::kFLOAT);

		nvinfer1::ILayer* add_maxpool(int layer_idx, darknet::Block& block, nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network);

		nvinfer1::ILayer* add_conv_bn_leaky(
			int layer_idx,
			darknet::Block& block,
			std::vector<float>& weight,
			std::vector<nvinfer1::Weights>& trt_weights,
			int& weight_ptr,
			int& input_channels,
			nvinfer1::ITensor* input,
			nvinfer1::INetworkDefinition* network
		);

		nvinfer1::ILayer* add_conv_linear(
			int layer_idx,
			darknet::Block& block,
			std::vector<float>& weight,
			std::vector<nvinfer1::Weights>& trt_weights,
			int& weight_ptr,
			int& input_channels,
			nvinfer1::ITensor* input,
			nvinfer1::INetworkDefinition* network
		);

		nvinfer1::ILayer* add_upsample(
			int layer_idx,
			darknet::Block& block,
			std::vector<float&> weights,
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
			nvinfer1::INetworkDefinition* network
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
	};
}

#endif
