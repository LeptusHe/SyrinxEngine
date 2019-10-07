#include "SyrinxCudaTexture.h"
#include <Common/SyrinxAssert.h>
#include <Exception/SyrinxException.h>
#include <Logging/SyrinxLogManager.h>
#include "SyrinxRadianceAssert.h"

namespace Syrinx {

cudaChannelFormatDesc GetChannelFormatDesc(PixelFormat format)
{
    switch (format) {
        case PixelFormat::RGB8:  return cudaCreateChannelDesc<uchar3>();
        case PixelFormat::RGBA8: return cudaCreateChannelDesc<uchar4>();
        case PixelFormat::RGBAF: return cudaCreateChannelDesc<float4>();
        default: {
            SYRINX_FAULT_FMT("fail to convert pixel format [{}] to cudaChannelDesc", format._to_string());
            SHOULD_NOT_GET_HERE();
        }
    }
}


cudaTextureAddressMode GetCudaTextureAddressMode(TextureWrapMethod wrapMethod)
{
    switch (wrapMethod) {
        case TextureWrapMethod::CLAMP_TO_BORDER: return cudaAddressModeBorder;
        case TextureWrapMethod::CLAMP_TO_EDGE:   return cudaAddressModeClamp;
        case TextureWrapMethod::MIRROR_CLAMP_TO_EDGE: return cudaAddressModeMirror;
        case TextureWrapMethod::REPEAT: return cudaAddressModeWrap;
        case TextureWrapMethod::MIRRORED_REPEAT: return cudaAddressModeWrap;
        default: {
            SYRINX_FAULT_FMT("fail to convert texture warp mode [{}] to cudaTextureAddressMode", wrapMethod._to_string());
            SHOULD_NOT_GET_HERE();
        }
    }
}


cudaTextureFilterMode GetCudaTextureFilterMode(TextureMinFilterMethod filterMethod)
{
    switch (filterMethod) {
        case TextureMinFilterMethod::NEAREST: return cudaFilterModePoint;
        case TextureMinFilterMethod::LINEAR: return cudaFilterModeLinear;
        case TextureMinFilterMethod::NEAREST_MIPMAP_NEAREST: return cudaFilterModePoint;
        case TextureMinFilterMethod::NEAREST_MIPMAP_LINEAR: return cudaFilterModeLinear;
        case TextureMinFilterMethod::LINEAR_MIPMAP_NEAREST: return cudaFilterModePoint;
        case TextureMinFilterMethod::LINEAR_MIPMAP_LINEAR: return cudaFilterModeLinear;
        default: {
            SYRINX_FAULT_FMT("fail to convert filter mode [{}] to cudaTextureFilterMode", filterMethod._to_string());
            SHOULD_NOT_GET_HERE();
        }
    }
}


cudaTextureFilterMode GetCudaTextureFilterMode(TextureMagFilterMethod filterMethod)
{
    switch (filterMethod) {
        case TextureMagFilterMethod::NEAREST: return cudaFilterModePoint;
        case TextureMagFilterMethod::LINEAR: return cudaFilterModeLinear;
        default: {
            SYRINX_FAULT_FMT("fail to convert filter mode [{}] to cudaTextureFilterMode", filterMethod._to_string());
            SHOULD_NOT_GET_HERE();
        }
    }
}


cudaTextureFilterMode GetCudaMipmapFilterMode(TextureMinFilterMethod filterMethod)
{
    switch (filterMethod) {
        case TextureMinFilterMethod::NEAREST_MIPMAP_NEAREST:
        case TextureMinFilterMethod::NEAREST_MIPMAP_LINEAR: return cudaFilterModePoint;
        case TextureMinFilterMethod::LINEAR_MIPMAP_NEAREST:
        case TextureMinFilterMethod::LINEAR_MIPMAP_LINEAR: return cudaFilterModeLinear;
        default: {
            SYRINX_FAULT_FMT("fail to convert filter mode [{}] to cudaTextureFilterMode", filterMethod._to_string());
            SHOULD_NOT_GET_HERE();
        }
    }
}



inline int GetMaxMipMapLevel(size_t width, size_t height)
{
    int  maxSize = static_cast<int>(std::max(width, height));
    int maxLevel = static_cast<int>(std::floor(std::log2(maxSize)));
    SYRINX_ENSURE(maxLevel >= 0);
    return maxLevel;
}



CudaTexture::CudaTexture(const std::string& name)
    : mName(name)
{
    SYRINX_ENSURE(!mName.empty());
}


CudaTexture::CudaTexture(CudaTexture&& rhs) noexcept
    : mName(std::move(rhs.mName))
    , mHandle(rhs.mHandle)
    , mWidth(rhs.mWidth)
    , mHeight(rhs.mHeight)
    , mDepth(rhs.mDepth)
    , mSizeInBytes(rhs.mSizeInBytes)
    , mLevelCount(rhs.mLevelCount)
    , mType(rhs.mType)
    , mFormat(rhs.mFormat)
    , mSamplingSetting(rhs.mSamplingSetting)
{
    SYRINX_ENSURE(rhs.mName.empty());
    rhs.mHandle = 0;
    rhs.mWidth = 0;
    rhs.mHeight = 0;
    rhs.mDepth = 1;
    rhs.mSizeInBytes = 0;
    rhs.mLevelCount = 0;
    rhs.mType = TextureType::TEXTURE_2D;
    rhs.mFormat = PixelFormat::RGBA8;
    rhs.mSamplingSetting = SamplingSetting();
}


void CudaTexture::enableMipmap(bool enable)
{
    mLevelCount = enable ? AutoGenerateMipmap : 1;
}


void CudaTexture::create()
{
    if (!isValid()) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "fail to create CUDA texture [{}]", mName);
    }

    cudaChannelFormatDesc channelFormatDesc = GetChannelFormatDesc(mFormat);
    mSizeInBytes = getSize(channelFormatDesc);
    SYRINX_CUDA_ASSERT(cudaMallocArray(&mMemory, &channelFormatDesc, mWidth, mHeight));

    cudaResourceDesc resourceDesc = {};
    resourceDesc.resType = cudaResourceTypeArray;
    resourceDesc.res.array.array = mMemory;

    cudaTextureDesc textureDesc = {};
    textureDesc.addressMode[0] = GetCudaTextureAddressMode(mSamplingSetting.getWrapSMethod());
    textureDesc.addressMode[1] = GetCudaTextureAddressMode(mSamplingSetting.getWrapTMethod());
    textureDesc.addressMode[2] = GetCudaTextureAddressMode(mSamplingSetting.getWrapRMethod());
    textureDesc.filterMode = GetCudaTextureFilterMode(mSamplingSetting.getMagFilterMethod());
    textureDesc.readMode = cudaReadModeNormalizedFloat;
    textureDesc.normalizedCoords = 1;
    textureDesc.maxAnisotropy = 1;
    textureDesc.maxMipmapLevelClamp = static_cast<float>(GetMaxMipMapLevel(mWidth, mHeight));
    textureDesc.minMipmapLevelClamp = 0;
    textureDesc.mipmapFilterMode = GetCudaMipmapFilterMode(mSamplingSetting.getMinFilterMethod());

    auto borderColor = mSamplingSetting.getBorderColor();
    textureDesc.borderColor[0] = borderColor.red();
    textureDesc.borderColor[1] = borderColor.green();
    textureDesc.borderColor[2] = borderColor.blue();
    textureDesc.borderColor[3] = borderColor.alpha();

    textureDesc.sRGB = 0;

    SYRINX_CUDA_ASSERT(cudaCreateTextureObject(&mHandle, &resourceDesc, &textureDesc, nullptr));
    SYRINX_ENSURE(mHandle != 0);
}


void CudaTexture::upload(uint8_t *src, size_t xOffset, size_t yOffset, size_t width, size_t height)
{
    SYRINX_EXPECT(src);
    SYRINX_EXPECT(xOffset + width <= mWidth);
    SYRINX_EXPECT(yOffset + height <= mHeight);

    size_t pitch = mSizeInBytes / mHeight;
    auto source = reinterpret_cast<void*>(src);
    SYRINX_CUDA_ASSERT(cudaMemcpy2DToArray(mMemory, xOffset, yOffset, source, pitch, width, height, cudaMemcpyHostToDevice));
}


void CudaTexture::upload(uint8_t *src, size_t width, size_t height)
{
    upload(src, 0, 0, width, height);
}


bool CudaTexture::isValid() const
{
    if (mWidth == 0 || mHeight == 0 || mDepth == 0)
        return false;

    if (mType._to_index() == TextureType::TEXTURE_2D && mDepth != 1)
        return false;

    if (mType._to_index() != TextureType::TEXTURE_2D && mDepth == 1)
        return false;

    return true;
}


size_t CudaTexture::getSize(cudaChannelFormatDesc channelFormatDesc)
{
    int bytesOfChannels = (channelFormatDesc.x + channelFormatDesc.y + channelFormatDesc.z + channelFormatDesc.w) / 8;
    return mHeight * mWidth * static_cast<size_t>(bytesOfChannels);
}

} // namespace Syrinx
