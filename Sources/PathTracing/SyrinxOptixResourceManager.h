#pragma once
#include <map>
#include <memory>
#include <optix.h>
#include <cuda_runtime.h>
#include <Container/SyrinxCache.h>
#include <Image/SyrinxImageReader.h>
#include <FileSystem/SyrinxFileManager.h>
#include "SyrinxCudaTexture.h"
#include "SyrinxCudaBuffer.h"
#include "SyrinxOptixContext.h"

namespace Syrinx {

class OptixResourceManager {
public:
    struct ProgramInfo {
        ProgramInfo() = default;
        ProgramInfo(const std::string& fileName, const std::string& entryFunctionName)
            : fileName(fileName)
            , entryFunctionName(entryFunctionName)
        {}

        bool isValid() const
        {
            return (!fileName.empty()) && (!entryFunctionName.empty());
        }

        bool operator==(const ProgramInfo& rhs) const
        {
            return (fileName == rhs.fileName) && (entryFunctionName == rhs.entryFunctionName);
        }

        std::string fileName;
        std::string entryFunctionName;
    };

    using BufferCache = Cache<std::string, CudaBuffer>;
    using TextureCache = Cache<std::string, CudaTexture>;
    using ModuleCache = std::map<std::string, OptixModule>;
    using ProgramGroupCache = std::map<std::string, OptixProgramGroup>;
    using PipelineCache = std::map<std::string, OptixPipeline>;

public:
    explicit OptixResourceManager(OptixContext *optixContext, FileManager *fileManager, std::unique_ptr<ImageReader>&& imageReader = std::make_unique<ImageReader>());
    virtual CudaBuffer* createBuffer(const std::string& name, size_t size);
    virtual CudaBuffer* createBuffer(const std::string& name, uint8_t *source, size_t size);
    virtual CudaBuffer* findBuffer(const std::string& name);
    virtual CudaTexture* createTexture2D(std::string& fileFullPath, ImageFormat format, bool enableMipmap);
    virtual CudaTexture* findTexture(std::string& name) const;
    virtual OptixProgramGroup createRayGenProgramGroup(const std::string& name, const ProgramInfo& programInfo);
    virtual OptixProgramGroup createMissProgramGroup(const std::string& name, const ProgramInfo& programInfo);
    virtual OptixProgramGroup createHitProgramGroup(const std::string& name, const ProgramInfo& closestHitProgramInfo, const ProgramInfo& anyHitProgramInfo, const ProgramInfo& intersectProgramInfo);
    virtual OptixProgramGroup createCallableProgramGroup(const std::string& name, const ProgramInfo& directCallableProgramInfo, const ProgramInfo& continuationCallableProgramInfo);
    virtual OptixProgramGroup createExceptionProgramGroup(const std::string& name, const ProgramInfo& exceptionProgramInfo);
    virtual OptixProgramGroup findProgramGroup(const std::string& name) const;
    virtual OptixPipeline createPipeline(const std::string& name, const std::vector<OptixProgramGroup>& programGroupList);
    virtual OptixPipeline findPipeline(const std::string& name);
    virtual void destroy(CudaBuffer *buffer);
    virtual void destroy(CudaTexture *texture);

private:
    void initOptions();
    virtual OptixModule createOrRetrieveModule(const std::string& fileName);
    virtual OptixModule createModule(const std::string& filePath);
    virtual OptixModule findModule(const std::string& filePath) const;
    virtual OptixProgramGroup createProgramGroup(const std::string& name, const OptixProgramGroupDesc& desc);

private:
    OptixContext *mOptixContext;
    FileManager *mFileManager;
    std::unique_ptr<ImageReader> mImageReader;
    BufferCache mBufferCache;
    TextureCache mTextureCache;
    ModuleCache mModuleCache;
    ProgramGroupCache mProgramGroupCache;
    PipelineCache mPipelineCache;

public:
    OptixModuleCompileOptions moduleCompileOptions;
    OptixPipelineCompileOptions pipelineCompileOptions;
    OptixPipelineLinkOptions pipelineLinkOptions;
};

} // namespace Syrinx