#pragma once
#include <memory>
#include "SyrinxCudaTexture.h"
#include "SyrinxCudaBuffer.h"
#include <Container/SyrinxCache.h>
#include <Image/SyrinxImageReader.h>
#include <FileSystem/SyrinxFileManager.h>

namespace Syrinx {

class CudaResourceManager {
public:
    using BufferCache = Cache<std::string, CudaBuffer>;
    using TextureCache = Cache<std::string, CudaTexture>;

public:
    explicit CudaResourceManager(std::unique_ptr<ImageReader>&& imageReader = std::make_unique<ImageReader>());

    virtual CudaBuffer* createBuffer(const std::string& name, size_t size);
    virtual CudaBuffer* createBuffer(const std::string& name, uint8_t *source, size_t size);
    virtual CudaBuffer* findBuffer(const std::string& name);
    virtual CudaTexture* createTexture2D(std::string& fileFullPath, ImageFormat format, bool enableMipmap);
    virtual CudaTexture* findTexture(std::string& name) const;
    virtual void destroy(CudaBuffer *buffer);
    virtual void destroy(CudaTexture *texture);

private:
    std::unique_ptr<ImageReader> mImageReader;
    BufferCache mBufferCache;
    TextureCache mTextureCache;
};

} // namespace Syrinx