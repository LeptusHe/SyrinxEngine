#pragma once
#include <memory>
#include <string>
#include <utility>
#include <unordered_set>
#include <unordered_map>
#include <Image/SyrinxImage.h>
#include <Image/SyrinxImageReader.h>
#include <HardwareResource/SyrinxHardwareBuffer.h>
#include <HardwareResource/SyrinxHardwareVertexBuffer.h>
#include <HardwareResource/SyrinxHardwareIndexBuffer.h>
#include <HardwareResource/SyrinxProgramStage.h>
#include <HardwareResource/SyrinxProgramPipeline.h>
#include <HardwareResource/SyrinxVertexInputState.h>
#include <HardwareResource/SyrinxHardwareTexture.h>
#include <ResourceManager/SyrinxFileManager.h>

namespace Syrinx {

class HardwareResourceManager {
public:
    using HardwareBufferNameSet = std::unordered_set<std::string>;
    using HardwareVertexBufferMap = std::unordered_map<std::string, std::unique_ptr<HardwareVertexBuffer>>;
    using HardwareIndexBufferMap = std::unordered_map<std::string, std::unique_ptr<HardwareIndexBuffer>>;
    using HardwareTextureMap = std::unordered_map<std::string, std::unique_ptr<HardwareTexture>>;
    using ProgramStageMap = std::unordered_map<std::string, std::unique_ptr<ProgramStage>>;
    using ProgramPipelineMap = std::unordered_map<std::string, std::unique_ptr<ProgramPipeline>>;

public:
    explicit HardwareResourceManager(FileManager *fileManager, std::unique_ptr<ImageReader>&& imageReader = std::make_unique<ImageReader>());
    virtual ~HardwareResourceManager() = default;
    template <typename T> HardwareVertexBuffer* createVertexBuffer(const std::string& name, size_t numVertex, size_t vertexSize, const T *data);
    template <typename T> HardwareIndexBuffer* createIndexBuffer(const std::string& name, size_t numIndex, IndexType indexType, const T *data);
    virtual HardwareVertexBuffer* findHardwareVertexBuffer(const std::string& name) const;
    virtual HardwareIndexBuffer* findHardwareIndexBuffer(const std::string& name) const;
    virtual HardwareTexture* createTexture(const std::string& fileName, ImageFormat format);
    virtual HardwareTexture* findTexture(const std::string& name);
    virtual ProgramStage* createProgramStage(const std::string& fileName, ProgramStageType stageType);
    virtual ProgramStage* findProgramStage(const std::string& name) const;
    virtual ProgramPipeline* createProgramPipeline(const std::string& name);
    virtual ProgramPipeline* findProgramPipeline(const std::string& name) const;

protected:
    virtual HardwareVertexBuffer* createVertexBuffer(const std::string& name, size_t numVertex, size_t vertexSize, const void *data);
    virtual HardwareIndexBuffer* createIndexBuffer(const std::string& name, size_t numIndex, IndexType indexType, const void *data);
    virtual std::unique_ptr<HardwareBuffer> createHardwareBuffer(const std::string& name);
    virtual void addProgramStage(ProgramStage *programStage);
    virtual void addProgramPipeline(ProgramPipeline* programPipeline);
    virtual void addHardwareTexture(HardwareTexture* hardwareTexture);
    virtual std::pair<bool, ProgramStage*> programStageExist(const std::string& fileName, ProgramStageType type) const;

private:
    FileManager *mFileManager;
    std::unique_ptr<ImageReader> mImageReader;
    HardwareBufferNameSet mHardwareBufferNameSet;
    HardwareVertexBufferMap mHardwareVertexBufferMap;
    HardwareIndexBufferMap mHardwareIndexBufferMap;
    HardwareTextureMap mHardwareTextureMap;
    ProgramStageMap mProgramStageMap;
    ProgramPipelineMap mProgramPipelineMap;
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