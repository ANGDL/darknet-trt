#ifndef _YOLO_H_
#define  _YOLO_H_

#include <memory>
#include "darknet_cfg.h"

namespace darknet {

	class Yolo {
	public:
		std::shared_ptr<NetConfig> config;
	};
}

#endif
