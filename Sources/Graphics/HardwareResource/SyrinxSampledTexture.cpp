#include "SyrinxSampledTexture.h"

namespace Syrinx {

SampledTexture::SampledTexture(const HardwareTextureView& textureView, const HardwareSampler& textureSampler)
    : mTextureView(textureView)
    , mTextureSampler(textureSampler)
{

}


const HardwareTextureView& SampledTexture::getTextureView() const
{
    return mTextureView;
}



const HardwareSampler& SampledTexture::getSampler() const
{
    return mTextureSampler;
}

} // namespace Syrinx
