#pragma once
#include <unordered_map>
#include <Math/SyrinxMath.h>
#include "RenderResource/SyrinxDepthTexture.h"
#include "HardwareResource/SyrinxHardwareResource.h"

namespace Syrinx {

class RenderTarget : public HardwareResource {
public:
    using RenderTextureBindingPoint = uint8_t;
    using RenderTextureMap = std::unordered_map<RenderTextureBindingPoint, const RenderTexture*>;

    static constexpr float DEFAULT_CLEAR_DEPTH_VALUE = 1.0;
    static const Color DEFAULT_CLEAR_COLOR_VALUE;

public:
    explicit RenderTarget(const std::string& name);
    ~RenderTarget() override = default;

    void addRenderTexture(RenderTextureBindingPoint bindingPoint, const RenderTexture *renderTexture);
    void addDepthTexture(const DepthTexture *depthTexture);
    bool create() override;
    void setClearColorValue(const Color& color);
    void setClearDepthValue(float depthValue);
    void clearDepthTexture();
    void clearRenderTexture();

private:
    bool isValidStatus();
    bool isValidToCreate() const override;

private:
    RenderTextureMap mRenderTextureMap;
    const DepthTexture *mDepthTexture;
    Color mClearColor;
    float mClearDepthValue;
};

} // namespace Syrinx