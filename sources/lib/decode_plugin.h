#ifndef _DECODE_PLUGIN_H_
#define _DECODE_PLUGIN_H_

#include "NvInferPlugin.h"
#include "darknet_utils.h"
#include "decode_kernel.cuh"

namespace darknet {
    using namespace nvinfer1;

    class DecodePlugin : public nvinfer1::IPluginV2Ext {
    private:
        float score_thresh_;
        std::vector<float> anchors_;

        float stride_;

        size_t grid_size_;
        size_t num_anchors_;
        size_t num_classes_;

    protected:
        void deserialize(void const *data, size_t length) {
            const char *d = static_cast<const char *>(data);
            read(d, score_thresh_);
            size_t anchors_size;
            read(d, anchors_size);
            while (anchors_size--) {
                float val;
                read(d, val);
                anchors_.push_back(val);
            }

            read(d, stride_);

            read(d, grid_size_);
            read(d, num_anchors_);
            read(d, num_classes_);
        }

        void serialize(void *buffer) const override {
            char *d = static_cast<char *> (buffer);
            write(d, score_thresh_);
            write(d, anchors_.size());
            for (auto &val : anchors_) {
                write(d, val);
            }

            write(d, stride_);
            write(d, grid_size_);
            write(d, num_anchors_);
            write(d, num_classes_);
        }

        size_t getSerializationSize() const override {
            return sizeof(score_thresh_) + sizeof(size_t) +
                   sizeof(float) * anchors_.size() + sizeof(stride_) + sizeof(grid_size_) +
                   sizeof(num_anchors_) + sizeof(num_classes_);
        }

    public:
        DecodePlugin(float score_thresh, std::vector<float> const &anchors,
                     int stride, size_t grid_size, size_t num_anchors, size_t num_classes) :
                score_thresh_(score_thresh), anchors_(anchors),
                stride_(stride), grid_size_(grid_size), num_anchors_(num_anchors), num_classes_(num_classes) {

        }

        DecodePlugin(void const *data, size_t length) {
            this->deserialize(data, length);
        }

        const char *getPluginType() const override {
            return "YoloDecode";
        }

        const char *getPluginVersion() const override {
            return "1";
        }

        int getNbOutputs() const override {
            return 3;
        }

        Dims getOutputDimensions(int index,
                                 const Dims *inputs, int nbInputDims) override {
            assert(nbInputDims == 1);
            assert(index < this->getNbOutputs());
            return Dims3(num_anchors_ * grid_size_ * grid_size_ * (index == 1 ? 4 : 1), 1, 1);
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
                    stride_,
                    grid_size_,
                    num_anchors_,
                    num_classes_,
                    anchors_,
                    score_thresh_,
                    nullptr,
                    0,
                    nullptr
            );
        }

        int enqueue(int batchSize,
                    const void *const *inputs, void **outputs,
                    void *workspace, cudaStream_t stream) override {

            //size_t pred_size = (5 + num_classes) * grid_size * grid_size;
            //float* test_input;
            //cudaMallocHost(&test_input, pred_size * sizeof(float));
            //cudaMemcpy(test_input, inputs[0], pred_size*sizeof(float), cudaMemcpyDeviceToHost);

            //for (int i = 0; i < pred_size; ++i) {
            //	printf("%f ", test_input[i]);
            //}
            //printf("\n ");
            //cudaFreeHost(test_input);

            return cuda_decode_layer(
                    inputs,
                    outputs,
                    batchSize,
                    stride_,
                    grid_size_,
                    num_anchors_,
                    num_classes_,
                    anchors_,
                    score_thresh_,
                    workspace,
                    this->getWorkspaceSize(batchSize),
                    stream
            );
        }

        void destroy() override {
            delete this;
        };

        const char *getPluginNamespace() const override {
            return "";
        }

        void setPluginNamespace(const char *N) override {

        }

        // IPluginV2Ext Methods
        DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const {
            assert(index < 3);
            return DataType::kFLOAT;
        }

        bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted,
                                          int nbInputs) const {
            return false;
        }

        bool canBroadcastInputAcrossBatch(int inputIndex) const { return false; }

        void configurePlugin(const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs,
                             const DataType *inputTypes, const DataType *outputTypes, const bool *inputIsBroadcast,
                             const bool *outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) {
            assert(*inputTypes == nvinfer1::DataType::kFLOAT &&
                   floatFormat == nvinfer1::PluginFormat::kLINEAR);
            assert(nbInputs == 1);
            assert(nbOutputs == 3);
            assert(inputDims != nullptr && inputDims[0].nbDims == 3);
            assert(num_anchors_ * (5 + num_classes_) == inputDims[0].d[0]);
            assert(grid_size_ == inputDims[0].d[1]);
            assert(grid_size_ == inputDims[0].d[2]);
        }

        IPluginV2Ext *clone() const override {
            return new DecodePlugin(score_thresh_, anchors_, stride_, grid_size_, num_anchors_, num_classes_);
        }
    };

    class DecodePluginCreator : public IPluginCreator {
    public:
        DecodePluginCreator() {}

        const char *getPluginName() const override {
            return "YoloDecode";
        }

        const char *getPluginVersion() const override {
            return "1";
        }

        const char *getPluginNamespace() const override {
            return "";
        }

        IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override {
            return new DecodePlugin(serialData, serialLength);
        }

        void setPluginNamespace(const char *N) override {}

        const PluginFieldCollection *getFieldNames() override { return nullptr; }

        IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) override { return nullptr; }
    };

    REGISTER_TENSORRT_PLUGIN(DecodePluginCreator);
}

#endif
