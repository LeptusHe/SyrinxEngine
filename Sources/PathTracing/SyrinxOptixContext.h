#pragma once
#include <vector>
#include <optix.h>
#include <cuda_runtime.h>
#include <Common/SyrinxSingleton.h>
#include <Scene/SyrinxEntity.h>
#include "SyrinxAccelerateStructure.h"

namespace Syrinx {

class OptixContext : public Singleton<OptixContext> {
public:
    OptixContext() = default;
    ~OptixContext();

    AccelerateStructure buildAccelerateStructure(const std::vector<Entity*>& entityList);

protected:
    void init();

private:
    cudaStream_t mCudaStream;
    CUcontext mCudaContext;
    cudaDeviceProp mCudaDeviceProperties;

    OptixDeviceContext mOptixContext;
};

} // namespace Syrinx