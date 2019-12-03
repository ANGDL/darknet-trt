#include <assert.h>
#include <iostream>
#include <fstream>
#include "yolo.h"
#include "darknet_utils.h"

darknet::Yolo::Yolo(NetConfig* config, float confidence_thresh, float nms_thresh) :
	config(config),
	prob_thresh(confidence_thresh),
	nms_thresh(nms_thresh)
{

}

void darknet::Yolo::create_yolo_engine(const nvinfer1::DataType data_type /*= nvinfer1::DataType::kFLOAT*/)
{

}

nvinfer1::ILayer* darknet::Yolo::add_maxpool(int layer_idx, darknet::Block& block, nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
	assert(block.at("type") == "maxpool");
	assert(block.find("stride") != block.end());
	assert(block.find("size") != block.end());

	int win_size = stoi(block.at("size"));
	int stride = stoi(block.at("stride"));
	nvinfer1::IPoolingLayer* pool = network->addPooling(*input, nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW(win_size, win_size));
	pool->setStride(nvinfer1::DimsHW(stride, stride));
	std::string layer_name = "maxpool_" + std::to_string(layer_idx);
	pool->setName(layer_name.c_str());

	return pool;
}

nvinfer1::ILayer* darknet::Yolo::add_conv_bn_leaky(int layer_idx, darknet::Block& block, std::vector<float>& weight, std::vector<nvinfer1::Weights>& trt_weights, int& weight_ptr, int& input_channels, nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
	assert(block.at("type") == "convolutional");
	assert(block.find("batch_normalize") != block.end());
	assert(block.find("filters") != block.end());
	assert(block.find("size") != block.end());
	assert(block.find("stride") != block.end());
	assert(block.find("pad") != block.end());
	assert(block.find("activation") != block.end());

	int filters = stoi(block.at("filters"));
	int k_size = stoi(block.at("size"));
	int stride = stoi(block.at("stride"));
	int pad = stoi(block.at("pad"));

	pad = pad ? (k_size - 1) / 2 : 0;

	std::vector<float> bn_baises, bn_weights, bn_means, bn_vars;

	for (int i = 0; i < filters; ++i)
	{
		bn_baises.push_back(weight[weight_ptr++]);
	}

	for (int i = 0; i < filters; ++i)
	{
		bn_weights.push_back(weight[weight_ptr++]);
	}

	for (int i = 0; i < filters; ++i)
	{
		bn_means.push_back(weight[weight_ptr++]);
	}

	for (int i = 0; i < filters; ++i)
	{
		bn_vars.push_back(weight[weight_ptr++]);
	}

	nvinfer1::IConvolutionLayer* conv = add_conv(layer_idx, filters, k_size, stride, pad, weight, weight_ptr, input_channels, input, network);
	trt_weights.push_back(conv->getBiasWeights());
	trt_weights.push_back(conv->getKernelWeights());

	nvinfer1::IScaleLayer* bn = add_bn(layer_idx, filters, bn_baises, bn_weights, bn_means, bn_vars, conv->getOutput(0), network);
	trt_weights.push_back(bn->getShift());
	trt_weights.push_back(bn->getScale());
	trt_weights.push_back(bn->getPower());

	nvinfer1::IPluginLayer* leaky = add_leakyRelu(layer_idx, bn->getOutput(0), network);

	return leaky;
}

nvinfer1::ILayer* darknet::Yolo::add_conv_linear(int layer_idx, darknet::Block& block, std::vector<float>& weight, std::vector<nvinfer1::Weights>& trt_weights, int& weight_ptr, int& input_channels, nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
	assert(block.at("type") == "convolutional");
	assert(block.find("batch_normalize") == block.end());
	assert(block.find("filters") != block.end());
	assert(block.find("size") != block.end());
	assert(block.find("stride") != block.end());
	assert(block.find("pad") != block.end());
	assert(block.find("activation") != block.end());
	assert(block.at("activation") == "linear");

	int filters = stoi(block.at("filters"));
	int k_size = stoi(block.at("size"));
	int stride = stoi(block.at("stride"));
	int pad = stoi(block.at("pad"));

	pad = pad ? (k_size - 1) / 2 : 0;


	nvinfer1::IConvolutionLayer* conv = add_conv(layer_idx, filters, k_size, stride, pad, weight, weight_ptr, input_channels, input, network);

	trt_weights.push_back(conv->getBiasWeights());
	trt_weights.push_back(conv->getKernelWeights());

	return conv;
}

nvinfer1::ILayer* darknet::Yolo::add_upsample(int layer_idx, darknet::Block& block, std::vector<float&> weights, std::vector<nvinfer1::Weights>& trt_weights, int& weight_ptr, int& input_channels, nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
	assert(block.at("type") == "upsample");
	assert(block.find("stride") != block.end());
	nvinfer1::Dims input_dims = input->getDimensions();
	assert(input_dims.nbDims == 3);

	float stride = stof(block.at("stride"));

	nvinfer1::IPlugin* upsample = new UpsampleLayer(stride, input_dims);
	nvinfer1::IPluginLayer* upsample_layer = network->addPlugin(&input, 1, *upsample);

	std::string layer_name = "upsample_" + to_string(layer_idx);
	upsample_layer->setName(layer_name.c_str());
}

nvinfer1::IConvolutionLayer* darknet::Yolo::add_conv(int layer_idx, int filters, int kernel_size, int stride, int pad, std::vector<float>& weight, int& weight_ptr, int& input_channels, nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
	float* bias_buff = new float[filters];
	for (int i = 0; i < filters; ++i)
	{
		bias_buff[i] = weight[weight_ptr++];
	}

	size_t kernel_data_len = (size_t)kernel_size * kernel_size * filters * input_channels;
	float* weight_buff = new float[kernel_data_len];
	for (size_t i = 0; i < kernel_data_len; i++)
	{
		weight_buff[i] = weight[weight_ptr++];
	}

	nvinfer1::Weights conv_bias{ nvinfer1::DataType::kFLOAT, bias_buff, filters };
	nvinfer1::Weights conv_weights{ nvinfer1::DataType::kFLOAT, weight_buff, kernel_data_len };

	nvinfer1::IConvolutionLayer* conv = network->addConvolution(*input, filters, nvinfer1::DimsHW(kernel_size, kernel_size), conv_weights, conv_bias);
	conv->setStride(DimsHW(stride, stride));
	conv->setPadding(DimsHW(pad, pad));

	std::string layer_name = "conv_" + to_string(layer_idx);
	conv->setName(layer_name.c_str());

	return conv;
}

/*
bn:
						   x - mean					  γ                  mean*γ
		y = γ * ---------------- + β = x * ------  + β -  ------------
						  -----------					 var                  var
						\|    var^2

trt scale:

		y = ( x * scale + shift) ^ power
*/

nvinfer1::IScaleLayer* darknet::Yolo::add_bn(int layer_idx, int filters, std::vector<float>& bn_biases, std::vector<float>& bn_weights,
	std::vector<float>& bn_mean, std::vector<float>& bn_var, nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
	float* scale_buff = new float[filters];
	float* shift_buff = new float[filters];
	float* power_buff = new float[filters];

	for (int i = 0; i < filters; i++)
	{
		scale_buff[i] = bn_weights[i] / bn_var[i];
		shift_buff[i] = bn_biases[i] - (bn_mean[i] * bn_weights[i]) / bn_var[i];
		power_buff[i] = 1.0;
	}

	nvinfer1::Weights shift{ nvinfer1::DataType::kFLOAT, shift_buff, filters };
	nvinfer1::Weights scale{ nvinfer1::DataType::kFLOAT, scale_buff, filters };
	nvinfer1::Weights power{ nvinfer1::DataType::kFLOAT, power_buff, filters };

	nvinfer1::IScaleLayer* bn = network->addScale(*input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);

	std::string layer_name = "batch_norm_" + to_string(layer_idx);
	bn->setName(layer_name.c_str());

	return bn;
}

nvinfer1::IPluginLayer* darknet::Yolo::add_leakyRelu(int layer_idx, nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
	nvinfer1::IPlugin* leaky_relu = nvinfer1::plugin::createPReLUPlugin(0.1);
	nvinfer1::IPluginLayer* leaky = network->addPlugin(&input, 1, *leaky_relu);
	std::string layer_name = "leaky_relu_" + to_string(layer_idx);
	leaky->setName(layer_name.c_str());
	return leaky;
}
