#pragma once
#include "HardwareResource/SyrinxHardwareTextureView.h"
#include "HardwareResource/SyrinxHardwareSampler.h"

namespace Syrinx {

class SampledTexture {
public:
    SampledTexture(const HardwareTextureView& textureView, const HardwareSampler& textureSampler);
    const HardwareTextureView& getTextureView() const;
    const HardwareSampler& getSampler() const;

private:
    const HardwareTextureView& mTextureView;
    const HardwareSampler& mTextureSampler;
};


} // namespace Syrinx