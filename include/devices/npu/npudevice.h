//
// Created by weimch on 12/01/23.
//

#ifndef FASTLLM_NPUEVICE_H
#define FASTLLM_NPUDEVICE_H

#include "device.h"
#include "devices/cpu/cputhreadpool.h"

namespace fastllm {

class NpuDevice : BaseDevice {
   public:
    NpuDevice();

    bool Malloc(void **ret, size_t size);  // 分配尺寸为size的空间
    bool Free(void *ret);                  // 释放ret

    bool CopyDataToCPU(void *dst, void *src, size_t size);
    bool CopyDataFromCPU(void *dst, void *src, size_t size);

    int threads = 4;
    ThreadPool *threadPool = nullptr;
};

class NpuToFloat16 : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuToFloat32 : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuAttention : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams,
                 const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuEmbedding : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams,
                 const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuLayerNormOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuRMSNormOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuLinearOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams,
                 const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuSplitOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams,
                 const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuCatOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams,
                 const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuCatDirectOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuMatMulOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams,
                 const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuMatMulTransBOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams,
                 const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuSoftMaxOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuSiluOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuGeluNewOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuSwigluOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams,
                 const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuMulOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuMulToOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuAddToOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuAttentionMaskOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuAlibiMaskOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuTopKOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams,
                 const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuPermuteOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams,
                 const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuPermuteSelfOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuRotatePosition2DOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuNearlyRotatePosition2DOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuLlamaRotatePosition2DOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuRepeatPenaltyOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuApplyLognAttnOp : BaseOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuCopyKVCacheOp : BaseOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams,
                 const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuSplitBatchOp : BaseBatchOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams,
                 const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuCatBatchOp : BaseBatchOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams,
                 const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuMulBatchOp : BaseBatchOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuMatMulBatchOp : BaseBatchOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams,
                 const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuMatMulTransBBatchOp : BaseBatchOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams,
                 const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuSoftmaxBatchOp : BaseBatchOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuCatDirectBatchOp : BaseBatchOperator {
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

class NpuAttentionBatchOp : BaseBatchOperator {
    void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams,
                 const IntDict &intParams);
    void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
};

}  // namespace fastllm

#endif
