#pragma once
#include <memory>
#include <string>
#include <utility>
#include <unordered_set>
#include <unordered_map>
#include <Container/SyrinxCache.h>
#include <Image/SyrinxImage.h>
#include <Image/SyrinxImageReader.h>
#include "HardwareResource/SyrinxHardwareBuffer.h"
#include "HardwareResource/SyrinxHardwareVertexBuffer.h"
#include "HardwareResource/SyrinxHardwareIndexBuffer.h"
#include "HardwareResource/SyrinxProgramStage.h"
#include "HardwareResource/SyrinxProgramPipeline.h"
#include "HardwareResource/SyrinxVertexInputState.h"
#include "HardwareResource/SyrinxHardwareTexture.h"
#include "HardwareResource/SyrinxHardwareTextureView.h"
#include "HardwareResource/SyrinxRenderTarget.h"

namespace Syrinx {

class HardwareResourceManager {
public:
    using HardwareBufferNameSet = std::unordered_set<std::string>;
    using HardwareVertexBufferCache = Cache<std::string, HardwareVertexBuffer>;
    using HardwareIndexBufferCache = Cache<std::string, HardwareIndexBuffer>;
    using HardwareUniformBufferMap = std::unordered_map<std::string, std::unique_ptr<HardwareUniformBuffer>>;
    using HardwareTextureCache = Cache<std::string, HardwareTexture>;
    using HardwareTextureViewCache = Cache<std::string, HardwareTextureView>;
    using HardwareSamplerCache = Cache<std::string, HardwareSampler>;
    using ProgramStageMap = std::unordered_map<std::string, std::unique_ptr<ProgramStage>>;
    using ProgramPipelineMap = std::unordered_map<std::string, std::unique_ptr<ProgramPipeline>>;
    using RenderTargetCache = Cache<std::string, RenderTarget>;
    using VertexInputStateCache = Cache<std::string, VertexInputState>;

public:
    explicit HardwareResourceManager(std::unique_ptr<ImageReader>&& imageReader = std::make_unique<ImageReader>());
    virtual ~HardwareResourceManager() = default;
    template <typename T> HardwareVertexBuffer* createVertexBuffer(const std::string& name, size_t numVertex, size_t vertexSize, const T *data = nullptr);
    HardwareVertexBuffer* createVertexBuffer(const std::string& name, size_t numVertex, size_t vertexSize);
    template <typename T> HardwareIndexBuffer* createIndexBuffer(const std::string& name, size_t numIndex, IndexType indexType, const T *data);
    HardwareIndexBuffer* createIndexBuffer(const std::string& name, size_t numIndex, IndexType indexType);
    virtual HardwareUniformBuffer* createUniformBuffer(const std::string& name, size_t sizeInBytes);
    virtual HardwareVertexBuffer* findHardwareVertexBuffer(const std::string& name) const;
    virtual HardwareIndexBuffer* findHardwareIndexBuffer(const std::string& name) const;
    virtual HardwareTexture* createTexture(const std::string& fileFullPath, ImageFormat format, bool enableMipmap);
    virtual HardwareTexture* createTexture2D(const std::string& name, const PixelFormat& format, uint32_t width, uint32_t height, bool enableMipmap);
    virtual HardwareTextureView* createTextureView(const std::string& name, HardwareTexture *texture, const TextureViewDesc& viewDesc);
    virtual HardwareTexture* findTexture(const std::string& name) const;
    virtual HardwareTextureView* findTextureView(const std::string& name) const;
    virtual HardwareSampler* createSampler(const std::string& name, const SamplingSetting& samplingSetting);
    virtual HardwareSampler* findSampler(const std::string& name) const;
    virtual ProgramStage* createProgramStage(const std::string& name, std::vector<uint32_t>&& binarySource, ProgramStageType stageType);
    virtual ProgramStage* findProgramStage(const std::string& name) const;
    virtual ProgramPipeline* createProgramPipeline(const std::string& name);
    virtual ProgramPipeline* findProgramPipeline(const std::string& name) const;
    virtual RenderTarget* createRenderTarget(const std::string& name, const RenderTarget::Desc& desc, uint32_t width, uint32_t height);
    virtual VertexInputState* createVertexInputState(const std::string& name);
    virtual bool destroyHardwareVertexBuffer(const std::string& name);
    virtual bool destroyHardwareIndexBuffer(const std::string& name);
    virtual bool destroyHardwareTexture(const std::string& name);
    virtual bool destroyHardwareTextureView(const std::string& name);
    virtual bool destroyHardwareSampler(const std::string& name);

protected:
    virtual HardwareVertexBuffer* createVertexBuffer(const std::string& name, size_t numVertex, size_t vertexSize, const void *data);
    virtual HardwareIndexBuffer* createIndexBuffer(const std::string& name, size_t numIndex, IndexType indexType, const void *data);
    virtual std::unique_ptr<HardwareBuffer> createHardwareBuffer(const std::string& name);
    virtual void addProgramStage(ProgramStage *programStage);
    virtual void addProgramPipeline(ProgramPipeline* programPipeline);
    virtual std::pair<bool, ProgramStage*> programStageExist(const std::string& fileName, ProgramStageType type) const;


private:
    std::unique_ptr<ImageReader> mImageReader;
    HardwareBufferNameSet mHardwareBufferNameSet;
    HardwareVertexBufferCache mHardwareVertexBufferCache;
    HardwareIndexBufferCache mHardwareIndexBufferCache;
    HardwareUniformBufferMap mHardwareUniformBufferMap;
    HardwareTextureCache mHardwareTextureCache;
    HardwareTextureViewCache mHardwareTextureViewCache;
    HardwareSamplerCache mSamplerCache;
    ProgramStageMap mProgramStageMap;
    ProgramPipelineMap mProgramPipelineMap;
    RenderTargetCache mRenderTargetCache;
    VertexInputStateCache mVertexInputStateCache;
};


template <typename T>
HardwareVertexBuffer* HardwareResourceManager::createVertexBuffer(const std::string& name, size_t numVertex, size_t vertexSize, const T *data)
{
    const auto source = reinterpret_cast<const void*>(data);
    return createVertexBuffer(name, numVertex, vertexSize, source);
}


template <typename T>
HardwareIndexBuffer* HardwareResourceManager::createIndexBuffer(const std::string& name, size_t numIndex, IndexType indexType, const T *data)
{
    const auto source = reinterpret_cast<const void*>(data);
    return createIndexBuffer(name, numIndex, indexType, source);
}

} // namespace Syrinx