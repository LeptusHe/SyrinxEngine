#pragma once
#include <vector>
#include <optix.h>
#include <cuda_runtime.h>
#include <Common/SyrinxSingleton.h>
#include <Scene/SyrinxEntity.h>
#include "SyrinxAccelerateStructure.h"

namespace Syrinx {

class OptixContext : public Singleton<OptixContext> {
private:
    constexpr static size_t LOG_BUFFER_MAX_SIZE = 1024;
    static char LogBuffer[LOG_BUFFER_MAX_SIZE];

public:
    OptixContext() = default;
    ~OptixContext();

    void init();
    OptixModule createModule(const std::string& sourceCode,
                             const OptixModuleCompileOptions& moduleCompileOptions,
                             const OptixPipelineCompileOptions& pipelineCompileOptions);
    OptixProgramGroup createProgramGroup(const OptixProgramGroupDesc& programGroupDesc);
    OptixPipeline createPipeline(const std::vector<OptixProgramGroup>& programGroupList,
                                 const OptixPipelineCompileOptions& compileOptions,
                                 const OptixPipelineLinkOptions& linkOptions);
    AccelerateStructure buildAccelerateStructure(const std::vector<Entity*>& entityList);

// TODO: private
public:
    cudaStream_t mCudaStream;
    CUcontext mCudaContext;
    cudaDeviceProp mCudaDeviceProperties;
    OptixDeviceContext mOptixContext;
};

} // namespace Syrinx