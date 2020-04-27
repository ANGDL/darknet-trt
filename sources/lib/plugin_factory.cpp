#include <assert.h>
#include <algorithm>
#include "plugin_factory.h"


darknet::PluginFactory::PluginFactory()
{

}

nvinfer1::IPlugin* darknet::PluginFactory::createPlugin(const char* layerName, const void* serialData, size_t serialLength)
{
	if (std::string(layerName).find("leaky") != std::string::npos) {
		unique_ptr_nvplugin leaky = unique_ptr_nvplugin(nvinfer1::plugin::createPReLUPlugin(serialData, serialLength));
		leakyReLU_layers.push_back(std::move(leaky));
		return leakyReLU_layers.back().get();
	}
	else if (std::string(layerName).find("upsample") != std::string::npos) {
		unique_ptr_iplugin upsample = unique_ptr_iplugin(new UpsampleLayer(serialData, serialLength));
		upsample_layers.push_back(std::move(upsample));
		return upsample_layers.back().get();
	}
	else {
		std::cerr << "ERROR: Unrecognised layer : " << layerName << std::endl;
		assert(0);
		return nullptr;
	}
}

void darknet::PluginFactory::destroy()
{
	for_each(leakyReLU_layers.begin(), leakyReLU_layers.end(),
		[](unique_ptr_nvplugin& p) {
		p.reset();
	});

	for_each(upsample_layers.begin(), upsample_layers.end(),
		[](unique_ptr_iplugin& p) {
		p.reset();
	});
}
