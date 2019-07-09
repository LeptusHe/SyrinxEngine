#pragma once
#include "SyrinxRenderTexture.h"

namespace Syrinx {

class DepthTexture {
public:
    DepthTexture();
    DepthTexture(const std::string& name, const RenderTexture& depthTexture);
    DepthTexture& operator=(const DepthTexture& rhs);
    ~DepthTexture() = default;

    explicit operator bool() const;
    HardwareResource::ResourceHandle getHandle() const;
    const std::string& getName() const;
    void setName(const std::string& name);
    void setRenderTexture(const RenderTexture& renderTexture);

private:
    std::string mName;
    RenderTexture mDepthTexture;
};

} // namespace Syrinx