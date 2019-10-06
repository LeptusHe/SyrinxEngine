#pragma once
#include <cuda_runtime.h>
#include "SyrinxPixelFormat.h"
#include "SyrinxGraphicsEnums.h"
#include "HardwareResource/SyrinxHardwareSampler.h"

namespace Syrinx {

class CudaTexture {
public:
    static constexpr size_t AutoGenerateMipmap = 0;

public:
    CudaTexture(const std::string& name);
    CudaTexture(const CudaTexture&) = delete;
    CudaTexture(CudaTexture&& rhs) noexcept;
    ~CudaTexture() = default;

    void setWidth(size_t width) { mWidth = width; }
    void setHeight(size_t height) { mHeight = height; }
    void setDepth(size_t depth) { mDepth = depth; }
    void setType(TextureType type) { mType = type; }
    void setPixelFormat(PixelFormat pixelFormat) { mFormat = pixelFormat; }
    void setSamplingSetting(const SamplingSetting& samplingSetting) { mSamplingSetting = samplingSetting; }
    void enableMipmap(bool enable);
    void create();

    size_t getWidth() const { return mWidth; }
    size_t getHeight() const { return mHeight; }
    size_t getDepth() const { return mDepth; }
    size_t getType() const { return mType; }
    PixelFormat getPixelFormat() const { return mFormat; }
    void upload(uint8_t *src, size_t destOffset, size_t size);

private:
    bool isValid() const;

private:
    std::string mName;
    cudaTextureObject_t mHandle = 0;
    size_t mWidth = 0;
    size_t mHeight = 0;
    size_t mDepth = 0;
    size_t mLevelCount = 0;
    TextureType mType = TextureType::TEXTURE_2D;
    PixelFormat mFormat = PixelFormat::RGBA8;
    SamplingSetting mSamplingSetting;
};

} // namespace Syrinx