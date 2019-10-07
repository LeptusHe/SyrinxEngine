#include "SyrinxCudaResourceManager.h"

namespace Syrinx {

CudaResourceManager::CudaResourceManager(std::unique_ptr<ImageReader>&& imageReader)
    : mImageReader(std::move(imageReader))
    , mBufferCache()
    , mTextureCache()
{
    SYRINX_ENSURE(mImageReader);
}


CudaBuffer* CudaResourceManager::createBuffer(const std::string& name, size_t size)
{
    SYRINX_EXPECT(!name.empty());
    SYRINX_EXPECT(!findBuffer(name));
    auto cudaBuffer = new CudaBuffer(name);
    cudaBuffer->allocate(size);
    mBufferCache.add(name, cudaBuffer);
    SYRINX_ENSURE(findBuffer(name) == cudaBuffer);
    return cudaBuffer;
}


CudaBuffer* CudaResourceManager::createBuffer(const std::string& name, uint8_t *source, size_t size)
{
    SYRINX_EXPECT(source && size > 0);
    auto cudaBuffer = createBuffer(name, size);
    cudaBuffer->upload(source, size);
    return cudaBuffer;
}


CudaBuffer* CudaResourceManager::findBuffer(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    return mBufferCache.find(name);
}


CudaTexture* CudaResourceManager::createTexture2D(std::string& fileFullPath, ImageFormat format, bool enableMipmap)
{
    SYRINX_EXPECT(!fileFullPath.empty());
    SYRINX_EXPECT(!findTexture(fileFullPath));
    Image image = mImageReader->read(fileFullPath, format);
    auto cudaTexture = new CudaTexture(fileFullPath);
    cudaTexture->setType(TextureType::TEXTURE_2D);
    cudaTexture->setPixelFormat(PixelFormat::_from_string(format._to_string()));
    cudaTexture->setWidth(image.getWidth());
    cudaTexture->setHeight(image.getHeight());
    cudaTexture->setDepth(1);
    cudaTexture->enableMipmap(enableMipmap);

    SamplingSetting samplingSetting;
    samplingSetting.setWrapSMethod(TextureWrapMethod::REPEAT);
    samplingSetting.setWrapTMethod(TextureWrapMethod::REPEAT);
    samplingSetting.setWrapRMethod(TextureWrapMethod::REPEAT);
    samplingSetting.setMinFilterMethod(TextureMinFilterMethod::LINEAR_MIPMAP_LINEAR);
    samplingSetting.setMagFilterMethod(TextureMagFilterMethod::LINEAR);

    cudaTexture->setSamplingSetting(samplingSetting);
    cudaTexture->create();

    mTextureCache.add(fileFullPath, cudaTexture);
    SYRINX_ENSURE(findTexture(fileFullPath) == cudaTexture);
    return cudaTexture;
}


CudaTexture* CudaResourceManager::findTexture(std::string& name) const
{
    SYRINX_EXPECT(!name.empty());
    return mTextureCache.find(name);
}


void CudaResourceManager::destroy(CudaBuffer *buffer)
{
    SYRINX_EXPECT(buffer);
    mBufferCache.remove(buffer->getName());
}


void CudaResourceManager::destroy(CudaTexture *texture)
{
    SYRINX_EXPECT(texture);
    mTextureCache.remove(texture->getName());
}

} // namespace Syrinx
