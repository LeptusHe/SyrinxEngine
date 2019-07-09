#include "SyrinxHardwareResourceManager.h"
#include <Exception/SyrinxException.h>

namespace Syrinx {

HardwareResourceManager::HardwareResourceManager(std::unique_ptr<ImageReader>&& imageReader)
    : mImageReader(std::move(imageReader))
{
    SYRINX_ENSURE(mImageReader);
    SYRINX_ENSURE(!imageReader);
}


HardwareVertexBuffer* HardwareResourceManager::createVertexBuffer(const std::string& name, size_t numVertex, size_t vertexSize, const void *data)
{
    SYRINX_EXPECT(!name.empty());
    SYRINX_EXPECT((numVertex > 0) && (vertexSize > 0));

    if (mHardwareVertexBufferCache.find(name)) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                   "fail to create hardware vertex buffer [{}] because the name of vertex buffer exists", name);
    }

    auto hardwareBuffer = createHardwareBuffer("raw buffer of vertex buffer [" + name + "]");
    auto hardwareVertexBuffer = new HardwareVertexBuffer(std::move(hardwareBuffer));
    hardwareVertexBuffer->setVertexNumber(numVertex);
    hardwareVertexBuffer->setVertexSizeInBytes(vertexSize);
    hardwareVertexBuffer->initData(data);
    hardwareVertexBuffer->create();

    mHardwareVertexBufferCache.add(name, hardwareVertexBuffer);
    SYRINX_ENSURE(hardwareVertexBuffer->isCreated());
    SYRINX_ENSURE(mHardwareVertexBufferCache.find(name));
    return hardwareVertexBuffer;
}


HardwareIndexBuffer* HardwareResourceManager::createIndexBuffer(const std::string& name, size_t numIndex, IndexType indexType, const void *data)
{
    SYRINX_EXPECT(!name.empty());
    SYRINX_EXPECT(numIndex > 0);

    if (mHardwareIndexBufferCache.find(name)) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                   "fail to create hardware index buffer [{}] because the name of index buffer exists", name);
    }

    auto hardwareBuffer = createHardwareBuffer("raw buffer of index buffer [" + name + "]");
    auto hardwareIndexBuffer = new HardwareIndexBuffer(std::move(hardwareBuffer));
    hardwareIndexBuffer->setIndexNumber(numIndex);
    hardwareIndexBuffer->setIndexType(indexType);
    hardwareIndexBuffer->initData(data);
    hardwareIndexBuffer->create();

    mHardwareIndexBufferCache.add(name, hardwareIndexBuffer);
    SYRINX_ENSURE(hardwareIndexBuffer->isCreated());
    SYRINX_ENSURE(mHardwareIndexBufferCache.find(name));
    return hardwareIndexBuffer;
}


HardwareVertexBuffer* HardwareResourceManager::createVertexBuffer(const std::string& name, size_t numVertex, size_t vertexSize)
{
    return createVertexBuffer(name, numVertex, vertexSize, nullptr);
}


HardwareIndexBuffer* HardwareResourceManager::createIndexBuffer(const std::string& name, size_t numIndex, IndexType indexType)
{
    return createIndexBuffer(name, numIndex, indexType, nullptr);
}


HardwareUniformBuffer* HardwareResourceManager::createUniformBuffer(const std::string& name, size_t sizeInBytes)
{
    SYRINX_EXPECT(!name.empty());
    SYRINX_EXPECT(sizeInBytes > 0);
    if (mHardwareUniformBufferMap.find(name) != std::end(mHardwareUniformBufferMap)) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                   "fail to create hardware uniform buffer [{}] because the name of uniform buffer exist", name);
    }

    auto hardwareBuffer = createHardwareBuffer("raw buffer of uniform buffer [" + name + "]");
    auto hardwareUniformBuffer = new HardwareUniformBuffer(std::move(hardwareBuffer));
    hardwareUniformBuffer->setSize(sizeInBytes);
    hardwareUniformBuffer->initData<uint8_t>(nullptr);
    hardwareUniformBuffer->create();

    mHardwareUniformBufferMap.insert({name, std::unique_ptr<HardwareUniformBuffer>(hardwareUniformBuffer)});
    SYRINX_ENSURE(hardwareUniformBuffer->isCreated());
    return hardwareUniformBuffer;
}


HardwareVertexBuffer* HardwareResourceManager::findHardwareVertexBuffer(const std::string& name) const
{
    SYRINX_EXPECT(!name.empty());
    return mHardwareVertexBufferCache.find(name);
}


HardwareIndexBuffer* HardwareResourceManager::findHardwareIndexBuffer(const std::string& name) const
{
    SYRINX_EXPECT(!name.empty());
    return mHardwareIndexBufferCache.find(name);
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


HardwareTexture* HardwareResourceManager::createTexture(const std::string& fileFullPath, ImageFormat format, bool enableMipmap)
{
    SYRINX_EXPECT(!fileFullPath.empty());
    Image image = mImageReader->read(fileFullPath, format);
    auto hardwareTexture = new HardwareTexture(fileFullPath);
    hardwareTexture->setType(TextureType::TEXTURE_2D);
    hardwareTexture->setPixelFormat(PixelFormat::_from_string(format._to_string()));
    hardwareTexture->setWidth(image.getWidth());
    hardwareTexture->setHeight(image.getHeight());
    hardwareTexture->enableMipmap(enableMipmap);
    hardwareTexture->create();
    hardwareTexture->write(image.getData(), image.getWidth(), image.getHeight());
    if (enableMipmap) {
        hardwareTexture->generateMipMap();
    }
    mHardwareTextureCache.add(fileFullPath, hardwareTexture);
    SYRINX_ENSURE(hardwareTexture->isCreated());
    SYRINX_ENSURE(findTexture(fileFullPath) == hardwareTexture);
    return hardwareTexture;
}


HardwareTexture* HardwareResourceManager::createTexture2D(const std::string& name, const PixelFormat& format, uint32_t width, uint32_t height, bool enableMipmap)
{
    SYRINX_EXPECT(!name.empty());
    SYRINX_EXPECT(width > 0 && height > 0);
    auto hardwareTexture = new HardwareTexture(name);
    hardwareTexture->setType(TextureType::TEXTURE_2D);
    hardwareTexture->setPixelFormat(format);
    hardwareTexture->setWidth(width);
    hardwareTexture->setHeight(height);
    hardwareTexture->enableMipmap(enableMipmap);
    hardwareTexture->create();
    mHardwareTextureCache.add(name, hardwareTexture);
    SYRINX_ENSURE(hardwareTexture->isCreated());
    SYRINX_ENSURE(findTexture(name) == hardwareTexture);
    return hardwareTexture;
}


HardwareTextureView* HardwareResourceManager::createTextureView(const std::string& name, HardwareTexture *texture, const TextureViewDesc& viewDesc)
{
    SYRINX_EXPECT(texture && texture->isCreated());
    SYRINX_EXPECT(!name.empty());
    auto hardwareTextureView = new HardwareTextureView(name, texture, viewDesc);
    hardwareTextureView->create();
    mHardwareTextureViewCache.add(name, hardwareTextureView);
    SYRINX_ENSURE(hardwareTextureView->isCreated());
    SYRINX_ENSURE(findTextureView(name) == hardwareTextureView);
    return hardwareTextureView;
}


HardwareTexture* HardwareResourceManager::findTexture(const std::string& name) const
{
    SYRINX_EXPECT(!name.empty());
    return mHardwareTextureCache.find(name);
}


HardwareTextureView* HardwareResourceManager::findTextureView(const std::string& name) const
{
    SYRINX_EXPECT(!name.empty());
    return mHardwareTextureViewCache.find(name);
}


HardwareSampler* HardwareResourceManager::createSampler(const std::string& name, const SamplingSetting& samplingSetting)
{
    SYRINX_EXPECT(!name.empty());
    auto hardwareSampler = new HardwareSampler(name, samplingSetting);
    hardwareSampler->create();
    mSamplerCache.add(name, hardwareSampler);
    SYRINX_ENSURE(hardwareSampler->isCreated());
    SYRINX_ENSURE(findSampler(name) == hardwareSampler);
    return hardwareSampler;
}


HardwareSampler* HardwareResourceManager::findSampler(const std::string& name) const
{
    return mSamplerCache.find(name);
}


ProgramStage* HardwareResourceManager::createProgramStage(const std::string& name, std::vector<uint32_t>&& binarySource, ProgramStageType stageType)
{
    SYRINX_EXPECT(!name.empty());
    auto programStage = new ProgramStage(name, this);
    programStage->setType(stageType);
    programStage->setBinarySource(std::move(binarySource));
    programStage->create();
    addProgramStage(programStage);
    SYRINX_ENSURE(programStage->isCreated());
    SYRINX_ENSURE(findProgramStage(name) == programStage);
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


RenderTarget* HardwareResourceManager::createRenderTarget(const std::string& name, const RenderTarget::Desc& desc, uint32_t width, uint32_t height)
{
    SYRINX_EXPECT(!name.empty());
    SYRINX_EXPECT(width > 0 && height > 0);
    auto renderTarget = new RenderTarget(name);
    for (int i = 0; i < RenderTarget::getMaxColorAttachmentCount(); ++ i) {
        const auto format = desc.getColorAttachmentFormat(i);
        if (format._to_index() != PixelFormat::UNKNOWN) {
            const std::string colorAttachmentName = "color attachment [render target = " + name + ", index = " + std::to_string(i) + "]";
            auto colorAttachmentTexture = createTexture2D(colorAttachmentName, format, width, height, false);
            const std::string colorAttachmentViewName = "texture view [texture = " + colorAttachmentName + "]";
            auto colorAttachmentTextureView = createTextureView(colorAttachmentViewName, colorAttachmentTexture, TextureViewDesc());
            RenderTexture renderTexture("render texture [" + colorAttachmentName + "]", colorAttachmentTextureView);
            renderTarget->addRenderTexture(i, renderTexture);
        }
    }
    if (desc.getDepthStencilFormat()._to_index() != PixelFormat::UNKNOWN) {
        const std::string depthAttachmentName = "depth attachment [" + name + "]";
        auto depthAttachmentTexture = createTexture2D(depthAttachmentName, desc.getDepthStencilFormat(), width, height, false);
        const std::string depthAttachmentTextureViewName = "texture view [texture = " + depthAttachmentName + "]";
        auto depthAttachmentTextureView = createTextureView(depthAttachmentTextureViewName, depthAttachmentTexture, TextureViewDesc());
        const std::string depthTextureName = "render texture [" + depthAttachmentTextureViewName + "]";
        RenderTexture depthRenderTexture(depthTextureName, depthAttachmentTextureView);
        DepthTexture depthTexture(depthTextureName, depthRenderTexture);
        renderTarget->addDepthTexture(depthTexture);
    }
    renderTarget->create();
    mRenderTargetCache.add(name, renderTarget);
    return renderTarget;
}


VertexInputState* HardwareResourceManager::createVertexInputState(const std::string& name)
{
    auto vertexInputState = new VertexInputState(name);
    vertexInputState->create();
    mVertexInputStateCache.add(name, vertexInputState);
    return vertexInputState;
}


bool HardwareResourceManager::destroyHardwareVertexBuffer(const std::string& name)
{
    auto vertexBuffer = mHardwareVertexBufferCache.find(name);
    if (!vertexBuffer) {
        return false;
    }

    auto hardwareBufferName = vertexBuffer->getBuffer().getName();
    mHardwareBufferNameSet.erase(hardwareBufferName);
    return mHardwareVertexBufferCache.remove(name);
}


bool HardwareResourceManager::destroyHardwareIndexBuffer(const std::string& name)
{
    auto indexBuffer = mHardwareIndexBufferCache.find(name);
    if (!indexBuffer) {
        return false;
    }

    auto hardwareBufferName = indexBuffer->getBuffer().getName();
    mHardwareBufferNameSet.erase(hardwareBufferName);
    return mHardwareIndexBufferCache.remove(name);
}


bool HardwareResourceManager::destroyHardwareTexture(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    return mHardwareTextureCache.remove(name);
}


bool HardwareResourceManager::destroyHardwareTextureView(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    return mHardwareTextureViewCache.remove(name);
}


bool HardwareResourceManager::destroyHardwareSampler(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    return mSamplerCache.remove(name);
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


std::pair<bool, ProgramStage*> HardwareResourceManager::programStageExist(const std::string& fileName, ProgramStageType type) const
{
    auto programStage = findProgramStage(fileName);
    if (programStage && (programStage->getType() == type)) {
        return {true, programStage};
    }
    return {false, nullptr};
}

} // namespace Syrinx