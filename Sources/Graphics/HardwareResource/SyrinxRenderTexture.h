#pragma once
#include <Image/SyrinxImage.h>
#include "SyrinxHardwareTextureView.h"
#include "SyrinxHardwareTexture.h"

namespace Syrinx {

class RenderTexture {
public:
    RenderTexture();
    RenderTexture(const std::string& name, const HardwareTextureView *hardwareTexture);
    RenderTexture(const RenderTexture& rhs);
    RenderTexture(RenderTexture&& rhs) noexcept;

    explicit operator bool() const;
    RenderTexture& operator=(const RenderTexture& rhs);
    void setName(const std::string& name);
    void setTextureView(const HardwareTextureView *textureView);
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
    const HardwareTextureView *mHardwareTextureView;
};

} // namespace Syrinx