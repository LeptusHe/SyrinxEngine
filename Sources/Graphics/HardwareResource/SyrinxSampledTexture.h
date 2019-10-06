#pragma once
#include "HardwareResource/SyrinxHardwareTextureView.h"
#include "HardwareResource/SyrinxHardwareSampler.h"

namespace Syrinx {

class SampledTexture {
public:
    SampledTexture() = default;
    SampledTexture(const HardwareTextureView *textureView, const HardwareSampler *textureSampler);
    SampledTexture(const SampledTexture& rhs) = default;
    ~SampledTexture() = default;

    operator bool() const;
    const HardwareTextureView& getTextureView() const;
    const HardwareSampler& getSampler() const;

private:
    const HardwareTextureView *mTextureView = nullptr;
    const HardwareSampler *mTextureSampler = nullptr;
};


} // namespace Syrinx