#pragma once
#include <cuda_runtime.h>
#include <Exception/SyrinxException.h>

#define SYRINX_CUDA_ASSERT(cudaFunc)                                                                                   \
{                                                                                                                      \
    cudaError_t errorCode = cudaFunc;                                                                                  \
    if (errorCode != cudaSuccess) {                                                                                    \
        const char *errorName = cudaGetErrorName(errorCode);                                                           \
        const char *errorMessage = cudaGetErrorString(errorCode);                                                      \
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::CUDAError, "CUDA ERROR: [{}] ({})", errorName, errorMessage);        \
    }                                                                                                                  \
}                                                                                                                      \
