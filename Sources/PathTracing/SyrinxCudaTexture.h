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
    explicit CudaTexture(const std::string& name);
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

    std::string getName() const { return mName; }
    size_t getWidth() const { return mWidth; }
    size_t getHeight() const { return mHeight; }
    size_t getDepth() const { return mDepth; }
    size_t getType() const { return mType; }
    PixelFormat getPixelFormat() const { return mFormat; }
    void upload(uint8_t *src, size_t xOffset, size_t yOffset, size_t width, size_t height);
    void upload(uint8_t *src, size_t width, size_t height);

private:
    bool isValid() const;
    size_t getSize(cudaChannelFormatDesc channelFormatDesc);

private:
    std::string mName;
    cudaArray_t mMemory = nullptr;
    cudaTextureObject_t mHandle = 0;
    size_t mWidth = 0;
    size_t mHeight = 0;
    size_t mDepth = 1;
    size_t mSizeInBytes = 0;
    size_t mLevelCount = 0;
    TextureType mType = TextureType::TEXTURE_2D;
    PixelFormat mFormat = PixelFormat::RGBA8;
    SamplingSetting mSamplingSetting;
};

} // namespace Syrinx