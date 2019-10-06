#include "SyrinxSampledTexture.h"

namespace Syrinx {

SampledTexture::SampledTexture(const HardwareTextureView *textureView, const HardwareSampler *textureSampler)
    : mTextureView(textureView)
    , mTextureSampler(textureSampler)
{
    SYRINX_ENSURE(mTextureView && mTextureSampler);
}


SampledTexture::operator bool() const
{
    return mTextureView && mTextureSampler;
}


const HardwareTextureView& SampledTexture::getTextureView() const
{
    SYRINX_EXPECT(mTextureView);
    return *mTextureView;
}



const HardwareSampler& SampledTexture::getSampler() const
{
    SYRINX_EXPECT(mTextureSampler);
    return *mTextureSampler;
}

} // namespace Syrinx
