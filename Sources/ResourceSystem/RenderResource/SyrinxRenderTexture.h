#pragma once
#include "Image/SyrinxImage.h"
#include "HardwareResource/SyrinxHardwareTexture.h"

namespace Syrinx {

class RenderTexture {
public:
    RenderTexture(const std::string& name, const HardwareTexture *hardwareTexture);
    ~RenderTexture() = default;

    HardwareResource::ResourceHandle getHandle() const;
    PixelFormat getPixelFormat() const;
    bool isDepthTexture() const;
    const std::string& getName() const;
    uint32_t getWidth() const;
    uint32_t getHeight() const;
    uint32_t getDepth() const;
    Image getImage() const;
    Image getSubImage(int xOffset, int yOffset, int width, int height, int level = 0) const;

private:
    std::string mName;
    const HardwareTexture *mHardwareTexture;
};

} // namespace Syrinx