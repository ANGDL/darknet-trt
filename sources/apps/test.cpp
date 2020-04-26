#include <iostream>
#include <filesystem>
#include <chrono>
#include "../lib/darknet_cfg.h"
#include "../lib/yolo.h"
#include "../lib/yolov3-tiny.h"
#include "../lib/yolov3.h"
#include "../lib/yolov3-nms.h"
#include "opencv2/opencv.hpp"

void test_parse_config()
{
	std::string curr_path{ std::filesystem::current_path().string() };
	std::string data_file = curr_path + "/config/coco.data";
	std::string cfg_file = curr_path + "/config/yolov3.cfg";
	std::string weights_file = curr_path + "/data/yolov3.weights";
	std::string calib_table_file = "";
	darknet::NetConfig cfg{ data_file, cfg_file, weights_file, calib_table_file, "kFLOAT" };
	cfg.display_blocks();
	std::cout << "PRECISION: " << cfg.PRECISION << std::endl;
	std::cout << "INPUT_BLOB_NAME: " << cfg.INPUT_BLOB_NAME << std::endl;
	std::cout << "INPUT_W: " << cfg.INPUT_W << std::endl;
	std::cout << "INPUT_H: " << cfg.INPUT_H << std::endl;
	std::cout << "INPUT_C: " << cfg.INPUT_C << std::endl;
	std::cout << "INPUT_SIZE: " << cfg.INPUT_SIZE << std::endl;
	std::cout << "OUTPUT_CLASSES: " << cfg.OUTPUT_CLASSES << std::endl;
	std::cout << "CLASS_NAMES: " << std::endl;
	for (auto& c : cfg.CLASS_NAMES)
	{
		std::cout << c << std::endl;
	}
}

void test_yolov3_config()
{
	std::string curr_path{ std::filesystem::current_path().string() };
	std::string data_file = curr_path + "/config/coco.data";
	std::string cfg_file = curr_path + "/config/yolov3.cfg";
	std::string calib_file = curr_path + "";
	std::string weights_file = curr_path + "";
	darknet::YoloV3Cfg cfg{ data_file, cfg_file, weights_file, calib_file, "kFLOAT" };

	std::cout << "BBOXES: " << cfg.BBOXES << std::endl;
	std::cout << "STRIDE_1: " << cfg.STRIDE_1 << std::endl;
	std::cout << "STRIDE_2: " << cfg.STRIDE_2 << std::endl;
	std::cout << "STRIDE_3: " << cfg.STRIDE_3 << std::endl;
	std::cout << "GRID_SIZE_1: " << cfg.GRID_SIZE_1 << std::endl;
	std::cout << "GRID_SIZE_2: " << cfg.GRID_SIZE_2 << std::endl;
	std::cout << "GRID_SIZE_3: " << cfg.GRID_SIZE_3 << std::endl;
	std::cout << "OUTPUT_SIZE_1: " << cfg.OUTPUT_SIZE_1 << std::endl;
	std::cout << "OUTPUT_SIZE_2: " << cfg.OUTPUT_SIZE_2 << std::endl;
	std::cout << "OUTPUT_SIZE_3: " << cfg.OUTPUT_SIZE_3 << std::endl;
	std::cout << "MASK_1: " << std::endl;
	for (auto& c : cfg.MASK_1)
	{
		std::cout << c << ", ";
	}
	std::cout << std::endl;

	std::cout << "MASK_2: " << std::endl;
	for (auto& c : cfg.MASK_2)
	{
		std::cout << c << ", ";
	}
	std::cout << std::endl;

	std::cout << "MASK_3: " << std::endl;
	for (auto& c : cfg.MASK_3)
	{
		std::cout << c << ", ";
	}
	std::cout << std::endl;



	std::cout << "OUTPUT_BLOB_NAME_1: " << cfg.OUTPUT_BLOB_NAME_1 << std::endl;
	std::cout << "OUTPUT_BLOB_NAME_2: " << cfg.OUTPUT_BLOB_NAME_2 << std::endl;
	std::cout << "OUTPUT_BLOB_NAME_3: " << cfg.OUTPUT_BLOB_NAME_3 << std::endl;


	std::cout << "ANCHORS: " << std::endl;
	for (auto& c : cfg.ANCHORS)
	{
		std::cout << c << ", ";
	}
	std::cout << std::endl;
}

void test_create_yolov3_engine() {
	std::string curr_path{ std::filesystem::current_path().string() };
	std::string data_file = curr_path + "/config/coco.data";
	std::string cfg_file = curr_path + "/config/yolov3.cfg";
	std::string weights_file = curr_path + "/data/yolov3.weights";
	std::string calib_table_file = "";
	darknet::NetConfig* cfg = darknet::DarkNetCfgFactory::create_network_config("yolov3", data_file, cfg_file, weights_file, calib_table_file, "kFLOAT");

	darknet::Yolo yolo_model(cfg, 8, 0.5, 0.5);
}

void test_create_yolov3_tiny_engine() {
	std::string curr_path{ std::filesystem::current_path().string() };
	std::string data_file = curr_path + "/config/coco.data";
	std::string cfg_file = curr_path + "/config/yolov3-tiny.cfg";
	std::string weights_file = curr_path + "/data/yolov3-tiny.weights";
	std::string calib_table_file = "";
	darknet::NetConfig* cfg = darknet::DarkNetCfgFactory::create_network_config("yolov3-tiny", data_file, cfg_file, weights_file, calib_table_file, "kFLOAT");

	darknet::Yolo yolo_model(cfg, 8, 0.5, 0.5);
}

void draw_img(const std::vector<BBoxInfo>& result, cv::Mat& img, const std::vector<cv::Scalar>& color, std::vector<std::string> class_names)
{
	int mark;
	int box_think = (img.rows + img.cols) * .001;
	float label_scale = img.rows * 0.0009;
	int base_line;
	for (const auto& item : result) {
		std::string label;
		std::stringstream stream;
		stream << class_names[item.label] << " " << item.prob << std::endl;
		std::getline(stream, label);

		auto size = cv::getTextSize(label, cv::FONT_HERSHEY_COMPLEX, label_scale, 1, &base_line);

		cv::rectangle(img, cv::Point(item.box.x1, item.box.y1),
			cv::Point(item.box.x2, item.box.y2),
			color[item.label], box_think, 8, 0);

		cv::putText(img, label,
			cv::Point(item.box.x2, item.box.y2 - size.height),
			cv::FONT_HERSHEY_COMPLEX, label_scale, color[item.label], box_think / 3, 8, 0);

	}

	cv::imshow("result", img);
	cv::waitKey(0);
}

cv::Mat prepare_image(cv::Mat& src, int input_w, int input_h) {

	if (src.channels() != 3)
	{
		std::cout << "Non RGB images are not supported : " << std::endl;
		assert(0);
	}

	cv::Mat marked_image;

	src.copyTo(marked_image);
	int height = src.rows;
	int width = src.cols;

	// resize the DsImage with scale
	float dim = std::max(height, width);
	int resize_h = ((height / dim) * input_h);
	int resize_w = ((width / dim) * input_w);
	float scale = static_cast<float>(resize_h) / static_cast<float>(height);

	// Additional checks for images with non even dims
	if ((input_w - resize_w) % 2) resize_w--;
	if ((input_h - resize_h) % 2) resize_h--;
	assert((input_w - resize_w) % 2 == 0);
	assert((input_h - resize_h) % 2 == 0);

	int x_offset = (input_w - resize_w) / 2;
	int y_offset = (input_h - resize_h) / 2;

	assert(2 * x_offset + resize_w == input_w);
	assert(2 * y_offset + resize_h == input_h);

	// resizing
	cv::resize(src, marked_image, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_CUBIC);
	// letterboxing
	cv::copyMakeBorder(marked_image, marked_image, y_offset, y_offset, x_offset,
		x_offset, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
	// converting to RGB
	cv::cvtColor(marked_image, marked_image, cv::COLOR_BGR2RGB);

	return marked_image;
}

cv::Mat blob_from_mats(const std::vector<cv::Mat>& input_images, const int& input_h,
	const int& input_w)
{
	std::vector<cv::Mat> letterboxStack(input_images.size());
	for (uint i = 0; i < input_images.size(); ++i)
	{
		input_images.at(i).copyTo(letterboxStack.at(i));
	}
	return cv::dnn::blobFromImages(letterboxStack, 1.0, cv::Size(input_w, input_h),
		cv::Scalar(0.0, 0.0, 0.0), false, false);
}


cv::Scalar randomColor(cv::RNG& rng) {
	int icolor = (unsigned)rng;
	return cv::Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

void test_yolov3_tiny_infer()
{
	const int batch_size = 1;
	std::string curr_path{ std::filesystem::current_path().string() };
	std::string data_file = curr_path + "/config/coco.data";
	std::string cfg_file = curr_path + "/config/yolov3-tiny.cfg";
	std::string weights_file = curr_path + "/data/yolov3-tiny.weights";
	std::string calib_table_file = "";
	darknet::NetConfig* cfg = darknet::DarkNetCfgFactory::create_network_config("yolov3-tiny", data_file, cfg_file, weights_file, calib_table_file, "kFLOAT");
	darknet::YoloV3Tiny net(cfg, batch_size, 0.5, 0.5);

	//cv::VideoCapture cap("D:/下载/东广场4#球机_东广场4#球机_20171121190000_20171121193000.avi");
	//if (!cap.isOpened())  // check if we succeeded
	//	return;

	uchar* input_buff = (uchar*)malloc(cfg->INPUT_SIZE * sizeof(float));
	if (input_buff == nullptr)
	{
		return;
	}

	uchar* p = input_buff;

	std::vector<cv::Scalar> color;
	for (int i = 0; i < cfg->OUTPUT_CLASSES; ++i) color.push_back(randomColor(cv::RNG(244)));

	std::vector<cv::Mat> frames;
	cv::Mat frame = cv::imread("person.jpg");
	//cv::imshow("frame", frame);
	//cv::waitKey(0);


	for (int i = 0; i < batch_size; ++i) {
		frames.push_back(frame.clone());
	}

	//input_mat = blob_from_mats(frames, net.net_cfg->INPUT_W, net.net_cfg->INPUT_H);

	//net.infer(input_mat.data);
	//for (int i = 0; i < frames.size(); ++i) {
	//	auto bboxes = net.get_detecions(i, frame.cols, frame.rows);
	//	draw_img(bboxes, frames[i], color, cfg->CLASS_NAMES);
	//}

	cv::Mat resized;
	cv::resize(frame, resized, cv::Size(net.net_cfg->INPUT_W, net.net_cfg->INPUT_H));
	cv::Mat resized_f;
	if (resized.depth() != CV_32FC3) {
		resized.convertTo(resized_f, CV_32FC3, 1.);
	}

	cv::imwrite("resized_f.jpg", resized);
	cv::imshow("resized_f", resized);
	cv::waitKey(0);

	for (int i = 0; i < batch_size; ++i) {
		memcpy(p, resized_f.data, net.net_cfg->INPUT_SIZE * sizeof(float));
		p += net.net_cfg->INPUT_SIZE * sizeof(float);
	}

	assert(p == input_buff + cfg->INPUT_SIZE * sizeof(float));

	net.infer(input_buff);
	for (int i = 0; i < frames.size(); ++i) {
		auto bboxes = net.get_detecions(i, frame.cols, frame.rows);
		draw_img(bboxes, frames[i], color, cfg->CLASS_NAMES);
	}

	delete[] input_buff;
	input_buff = nullptr;
}

void test_yolov3_infer()
{
	const int batch_size = 1;
	std::string curr_path{ std::filesystem::current_path().string() };
	std::string data_file = curr_path + "/config/coco.data";
	std::string cfg_file = curr_path + "/config/yolov3.cfg";
	std::string weights_file = curr_path + "/data/yolov3.weights";
	std::string calib_table_file = "";
	darknet::NetConfig* cfg = darknet::DarkNetCfgFactory::create_network_config("yolov3", data_file, cfg_file, weights_file, calib_table_file, "kHALF");
	darknet::YoloV3 net(cfg, batch_size, 0.5, 0.5);


	//cv::VideoCapture cap("D:/下载/东广场4#球机_东广场4#球机_20171121190000_20171121193000.avi");
	//if (!cap.isOpened())  // check if we succeeded
	//	return;


	std::vector<cv::Scalar> color;
	for (int i = 0; i < cfg->OUTPUT_CLASSES; ++i) color.push_back(randomColor(cv::RNG(244)));

	std::vector<cv::Mat> frames;

	cv::Mat frame = cv::imread("eagle.jpg");
	//cv::imshow("frame", frame);
	//cv::waitKey(0);

	cv::Mat input_mat = prepare_image(frame, net.net_cfg->INPUT_W, net.net_cfg->INPUT_H);

	for (int i = 0; i < batch_size; ++i) {
		frames.push_back(frame.clone());
	}

	input_mat = blob_from_mats(frames, net.net_cfg->INPUT_W, net.net_cfg->INPUT_H);

	auto start_time = std::chrono::system_clock::now();

	net.infer(input_mat.data);

	auto end_time = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_time = end_time - start_time;
	std::cout << "elapsed time: " << elapsed_time.count() * 1000 << "ms" << std::endl;

	for (int i = 0; i < frames.size(); ++i) {
		auto bboxes = net.get_detecions(i, frame.cols, frame.rows);
		//draw_img(bboxes, frames[i], color, cfg->CLASS_NAMES);
	}
}


void test_yolov3_nms()
{
	const int batch_size = 1;
	std::string curr_path{ std::filesystem::current_path().string() };
	std::string data_file = curr_path + "/config/coco.data";
	std::string cfg_file = curr_path + "/config/yolov3-tiny.cfg";
	std::string weights_file = curr_path + "/data/yolov3-tiny.weights";
	std::string calib_table_file = "";
	darknet::NetConfig* cfg = darknet::DarkNetCfgFactory::create_network_config("yolov3-tiny", data_file, cfg_file, weights_file, calib_table_file, "kFLOAT");

	darknet::YoloV3NMS net(cfg, batch_size, 0.5, 0.5);

	uchar* input_buff = (uchar*)malloc(cfg->INPUT_SIZE * sizeof(float));
	if (input_buff == nullptr)
	{
		return;
	}

	uchar* p = input_buff;

	std::vector<cv::Scalar> color;
	for (int i = 0; i < cfg->OUTPUT_CLASSES; ++i) color.push_back(randomColor(cv::RNG(244)));

	std::vector<cv::Mat> frames;
	cv::Mat frame = cv::imread("person.jpg");

	for (int i = 0; i < batch_size; ++i) {
		frames.push_back(frame.clone());
	}

	cv::Mat resized;
	cv::resize(frame, resized, cv::Size(cfg->INPUT_W, cfg->INPUT_H));
	cv::Mat resized_f;
	if (resized.depth() != CV_32FC3) {
		resized.convertTo(resized_f, CV_32FC3, 1.);
	}

	cv::imwrite("resized_f.jpg", resized);
	cv::imshow("resized_f", resized);
	cv::waitKey(0);

	for (int i = 0; i < batch_size; ++i) {
		memcpy(p, resized_f.data, cfg->INPUT_SIZE * sizeof(float));
		p += cfg->INPUT_SIZE * sizeof(float);
	}

	assert(p == input_buff + cfg->INPUT_SIZE * sizeof(float));

	net.infer(input_buff);
	auto bboxes = net.get_detecions(frame.cols, frame.rows);

	for (int i = 0; i < frames.size(); ++i) {
		draw_img(bboxes[i], frames[i], color, cfg->CLASS_NAMES);
	}

	free(input_buff);
	input_buff = nullptr;
}

int main()
{
	namespace fs = std::filesystem;
	std::cout << "Current path is " << fs::current_path() << '\n';
	//test_parse_config();
	//test_yolov3_config();
	//test_create_yolov3_engine();
	//test_create_yolov3_tiny_engine();
	//test_yolov3_tiny_infer();
	// test_yolov3_infer();
	test_yolov3_nms();
	return 0;
}
