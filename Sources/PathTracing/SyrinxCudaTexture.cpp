#include "SyrinxCudaTexture.h"
#include <Common/SyrinxAssert.h>
#include <Exception/SyrinxException.h>

namespace Syrinx {

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
    , mLevelCount(rhs.mLevelCount)
    , mType(rhs.mType)
    , mFormat(rhs.mFormat)
    , mSamplingSetting(rhs.mSamplingSetting)
{
    SYRINX_ENSURE(rhs.mName.empty());
    rhs.mHandle = 0;
    rhs.mWidth = 0;
    rhs.mHeight = 0;
    rhs.mDepth = 0;
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

    cudaResourceDesc resourceDesc = {};

    cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<float4>();
}

} // namespace Syrinx
