#pragma once
#include "SyrinxHardwareTexture.h"

namespace Syrinx {

struct TextureViewDesc {
    TextureType type = TextureType::TEXTURE_2D;
    uint32_t baseLevel = 0;
    uint32_t levelCount = 1;
    uint32_t baseLayer = 0;
    uint32_t layerCount = 1;
};


class HardwareTextureView : public HardwareResource {
public:
    HardwareTextureView(const std::string& name, HardwareTexture *texture, const TextureViewDesc& viewDesc = TextureViewDesc());
    bool create() override;
    PixelFormat getPixelFormat() const;
    uint32_t getWidth() const;
    uint32_t getHeight() const;
    uint32_t getDepth() const;

protected:
    bool isValidToCreate() const override;

private:
    HardwareTexture *mHardwareTexture;
    TextureViewDesc mViewDesc;
};

} // namespace Syrinx