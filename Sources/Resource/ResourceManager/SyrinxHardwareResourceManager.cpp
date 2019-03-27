#include "ResourceManager/SyrinxHardwareResourceManager.h"
#include <Exception/SyrinxException.h>

namespace Syrinx {

HardwareResourceManager::HardwareResourceManager(FileManager *fileManager, std::unique_ptr<ImageReader>&& imageReader)
    : mFileManager(fileManager)
    , mImageReader(std::move(imageReader))
{
    SYRINX_ENSURE(mFileManager);
    SYRINX_ENSURE(mImageReader);
    SYRINX_ENSURE(!imageReader);
}


HardwareVertexBuffer* HardwareResourceManager::createVertexBuffer(const std::string& name, size_t numVertex, size_t vertexSize, const void *data)
{
    SYRINX_EXPECT(!name.empty());
    SYRINX_EXPECT((numVertex > 0) && (vertexSize > 0));
    SYRINX_EXPECT(data);

    if (findHardwareVertexBuffer(name)) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                   "fail to create hardware vertex buffer [{}] because the name of vertex buffer exists", name);
    }

    auto hardwareBuffer = createHardwareBuffer("raw buffer of vertex buffer [" + name + "]");
    auto hardwareVertexBuffer = new HardwareVertexBuffer(std::move(hardwareBuffer));
    hardwareVertexBuffer->setVertexNumber(numVertex);
    hardwareVertexBuffer->setVertexSizeInBytes(vertexSize);
    hardwareVertexBuffer->setData(data);
    hardwareVertexBuffer->create();

    mHardwareVertexBufferMap.insert({std::string(name), std::unique_ptr<HardwareVertexBuffer>(hardwareVertexBuffer)});
    SYRINX_ENSURE(hardwareVertexBuffer->isCreated());
    SYRINX_ENSURE(findHardwareVertexBuffer(name));
    return hardwareVertexBuffer;
}


HardwareIndexBuffer* HardwareResourceManager::createIndexBuffer(const std::string& name, size_t numIndex, IndexType indexType, const void *data)
{
    SYRINX_EXPECT(!name.empty());
    SYRINX_EXPECT(numIndex > 0);
    SYRINX_EXPECT(data);

    if (findHardwareIndexBuffer(name)) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                   "fail to create hardware index buffer [{}] because the name of index buffer exists", name);
    }

    auto hardwareBuffer = createHardwareBuffer("raw buffer of index buffer [" + name + "]");
    auto hardwareIndexBuffer = new HardwareIndexBuffer(std::move(hardwareBuffer));
    hardwareIndexBuffer->setIndexNumber(numIndex);
    hardwareIndexBuffer->setIndexType(indexType);
    hardwareIndexBuffer->setData(data);
    hardwareIndexBuffer->create();

    mHardwareIndexBufferMap.insert({std::string(name), std::unique_ptr<HardwareIndexBuffer>(hardwareIndexBuffer)});
    SYRINX_ENSURE(hardwareIndexBuffer->isCreated());
    SYRINX_ENSURE(findHardwareIndexBuffer(name));
    return hardwareIndexBuffer;
}


std::unique_ptr<HardwareBuffer> HardwareResourceManager::createHardwareBuffer(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    if (mHardwareBufferNameSet.find(name) != std::end(mHardwareBufferNameSet)) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                   "fail to create hardware buffer [{}] because the name of hardware buffer exists", name);
    }
    mHardwareBufferNameSet.insert(name);
    SYRINX_ENSURE(mHardwareBufferNameSet.find(name) != std::end(mHardwareBufferNameSet));
    return std::make_unique<HardwareBuffer>(name);
}


HardwareVertexBuffer* HardwareResourceManager::findHardwareVertexBuffer(const std::string& name) const
{
    SYRINX_EXPECT(!name.empty());
    const auto iter = mHardwareVertexBufferMap.find(name);
    if (iter != std::end(mHardwareVertexBufferMap)) {
        return iter->second.get();
    }
    return nullptr;
}


HardwareIndexBuffer* HardwareResourceManager::findHardwareIndexBuffer(const std::string& name) const
{
    SYRINX_EXPECT(!name.empty());
    const auto iter = mHardwareIndexBufferMap.find(name);
    if (iter != std::end(mHardwareIndexBufferMap)) {
        return iter->second.get();
    }
    return nullptr;
}


HardwareTexture* HardwareResourceManager::createTexture(const std::string& fileName, ImageFormat format)
{
    SYRINX_EXPECT(!fileName.empty());
    auto [fileExist, filePath] = mFileManager->findFile(fileName);
    if (!fileExist) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileNotFound, "can not find file [{}]", fileName);
    }

    Image image = mImageReader->read(filePath, format);
    auto hardwareTexture = new HardwareTexture(fileName);
    hardwareTexture->setType(TextureType::TEXTURE_2D);
    hardwareTexture->setPixelFormat(PixelFormat::_from_string(format._to_string()));
    hardwareTexture->setWidth(image.getWidth());
    hardwareTexture->setHeight(image.getHeight());
    hardwareTexture->create();
    hardwareTexture->write(image.getData(), image.getWidth(), image.getHeight());
    addHardwareTexture(hardwareTexture);
    SYRINX_ENSURE(hardwareTexture->isCreated());
    SYRINX_ENSURE(findTexture(fileName) == hardwareTexture);
    return hardwareTexture;
}


HardwareTexture* HardwareResourceManager::findTexture(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    auto iter = mHardwareTextureMap.find(name);
    if (iter == std::end(mHardwareTextureMap)) {
        return nullptr;
    }
    return iter->second.get();
}


ProgramStage* HardwareResourceManager::createProgramStage(const std::string& fileName, ProgramStageType stageType)
{
    SYRINX_EXPECT(!fileName.empty());
    if (auto [exist, programStage] = programStageExist(fileName, stageType); exist) {
        return programStage;
    }

    auto fileStream = mFileManager->openFile(fileName, FileAccessMode::READ);
    if (!fileStream) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "can not open file [{}]", fileName);
    }

    auto programStage = new ProgramStage(fileName);
    programStage->setType(stageType);
    programStage->setSource(fileStream->getAsString());
    programStage->create();
    addProgramStage(programStage);
    SYRINX_ENSURE(programStage->isCreated());
    SYRINX_ENSURE(findProgramStage(fileName) == programStage);
    return programStage;
}


ProgramStage* HardwareResourceManager::findProgramStage(const std::string& name) const
{
    SYRINX_EXPECT(!name.empty());
    auto iter = mProgramStageMap.find(name);
    if (iter == std::end(mProgramStageMap)) {
        return nullptr;
    }
    return iter->second.get();
}


ProgramPipeline* HardwareResourceManager::createProgramPipeline(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    auto programPipeline = findProgramPipeline(name);
    if (programPipeline) {
        return programPipeline;
    }

    programPipeline = new ProgramPipeline(name);
    programPipeline->create();
    addProgramPipeline(programPipeline);
    SYRINX_ENSURE(programPipeline->isCreated());
    SYRINX_ENSURE(findProgramPipeline(name) == programPipeline);
    return programPipeline;
}


ProgramPipeline* HardwareResourceManager::findProgramPipeline(const std::string& name) const
{
    SYRINX_EXPECT(!name.empty());
    auto iter = mProgramPipelineMap.find(name);
    if (iter == std::end(mProgramPipelineMap)) {
        return nullptr;
    }
    return iter->second.get();
}


void HardwareResourceManager::addProgramStage(ProgramStage *programStage)
{
    SYRINX_EXPECT(programStage);
    SYRINX_EXPECT(!findProgramStage(programStage->getName()));
    mProgramStageMap[programStage->getName()] = std::unique_ptr<ProgramStage>(programStage);
    SYRINX_ENSURE(findProgramStage(programStage->getName()) == programStage);
}


void HardwareResourceManager::addProgramPipeline(ProgramPipeline *programPipeline)
{
    SYRINX_EXPECT(programPipeline);
    SYRINX_EXPECT(!findProgramPipeline(programPipeline->getName()));
    mProgramPipelineMap[programPipeline->getName()] = std::unique_ptr<ProgramPipeline>(programPipeline);
    SYRINX_ENSURE(findProgramPipeline(programPipeline->getName()) == programPipeline);
}


void HardwareResourceManager::addHardwareTexture(Syrinx::HardwareTexture *hardwareTexture)
{
    SYRINX_EXPECT(hardwareTexture);
    SYRINX_EXPECT(!findTexture(hardwareTexture->getName()));
    mHardwareTextureMap[hardwareTexture->getName()] = std::unique_ptr<HardwareTexture>(hardwareTexture);
    SYRINX_ENSURE(findTexture(hardwareTexture->getName()) == hardwareTexture);
}


std::pair<bool, ProgramStage*> HardwareResourceManager::programStageExist(const std::string& fileName, ProgramStageType type) const
{
    auto programStage = findProgramStage(fileName);
    if (programStage && (programStage->getType() == type)) {
        return {true, programStage};
    }
    return {false, nullptr};
}

} // namespace Syrinx