#pragma once
#include "RenderResource/SyrinxRenderTexture.h"

namespace Syrinx {

class DepthTexture {
public:
    DepthTexture(const std::string& name, const RenderTexture *depthTexture);
    ~DepthTexture() = default;

    HardwareResource::ResourceHandle getHandle() const;
    const std::string& getName() const;

private:
    std::string mName;
    const RenderTexture *mDepthTexture;
};

} // namespace Syrinx