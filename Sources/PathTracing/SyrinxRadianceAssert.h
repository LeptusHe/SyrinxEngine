#pragma once
#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <Exception/SyrinxException.h>

#define SYRINX_CUDA_ASSERT(cudaCall)                                                                                   \
{                                                                                                                      \
    cudaError_t errorCode = cudaCall;                                                                                  \
    if (errorCode != cudaSuccess) {                                                                                    \
        const char *errorName = cudaGetErrorName(errorCode);                                                           \
        const char *errorMessage = cudaGetErrorString(errorCode);                                                      \
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::CUDAError,                                                           \
            "CUDA ERROR: [{}] ({})", errorName, errorMessage);                                                         \
    }                                                                                                                  \
}                                                                                                                      \


#define SYRINX_OPTIX_ASSERT(optixCall)                                                                                 \
{                                                                                                                      \
    OptixResult result = optixCall;                                                                                    \
    if (result != OPTIX_SUCCESS) {                                                                                     \
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::CUDAError,                                                           \
            "OptiX call [{}] failed with error [{}]", #optixCall, optixGetErrorString(result));                        \
    }                                                                                                                  \
}                                                                                                                      \