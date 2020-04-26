#ifndef _PLAGIN_FACTORY_H_
#define _PLAGIN_FACTORY_H_
#include "yolo_layer_plugin.h"
#include "upsample_plugin.h"
#include "decode_plugin.h"
#include "nms_plugin.h"

namespace darknet {
	using namespace nvinfer1;


	// Plugin Factory
	class PluginFactory : public nvinfer1::IPluginFactory
	{
	public:
		PluginFactory();
		nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength);
		void destroy();

	private:

		// tensor rt官方的plugin deleter 
		struct INvPluginDeleter
		{
			void operator()(nvinfer1::plugin::INvPlugin* ptr) {
				if (ptr) {
					ptr->destroy();
				}
			}
		};

		// 自定义的plugin deleter
		struct IPluginDeleter
		{
			void operator()(nvinfer1::IPlugin* ptr) {
				ptr->terminate();
			}
		};

		typedef std::unique_ptr<nvinfer1::plugin::INvPlugin, INvPluginDeleter> unique_ptr_nvplugin;
		typedef std::unique_ptr<nvinfer1::IPlugin, IPluginDeleter> unique_ptr_iplugin;

		std::vector<unique_ptr_nvplugin> leakyReLU_layers;
		std::vector<unique_ptr_iplugin> yolo_layers;
		std::vector<unique_ptr_iplugin> upsample_layers;
	};
}


#endif
