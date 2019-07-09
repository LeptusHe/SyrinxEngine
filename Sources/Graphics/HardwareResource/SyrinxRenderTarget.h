#pragma once
#include <vector>
#include <unordered_map>
#include <Math/SyrinxMath.h>
#include "SyrinxDepthTexture.h"
#include "SyrinxHardwareResource.h"

namespace Syrinx {

class RenderTarget : public HardwareResource {
public:
    static int getMaxColorAttachmentCount();

public:
    class Desc {
    private:
        struct AttachmentDesc {
            PixelFormat pixelFormat = PixelFormat::UNKNOWN;
        };

    public:
        Desc();
        Desc& setColorAttachment(uint32_t index, const PixelFormat& format);
        Desc& setDepthStencilAttachment(const PixelFormat& format);
        PixelFormat getColorAttachmentFormat(uint32_t index) const;
        PixelFormat getDepthStencilFormat() const;

    private:
        std::vector<AttachmentDesc> mColorAttachmentDescList;
        AttachmentDesc mDepthStencilAttachmentDesc;
    };

public:
    using RenderTextureBindingPoint = uint8_t;
    using RenderTextureMap = std::unordered_map<RenderTextureBindingPoint, RenderTexture>;

    static constexpr float DEFAULT_CLEAR_DEPTH_VALUE = 1.0;
    static const Color DEFAULT_CLEAR_COLOR_VALUE;

public:
    explicit RenderTarget(const std::string& name);
    ~RenderTarget() override = default;

    const RenderTexture* getColorAttachment(RenderTextureBindingPoint bindingPoint) const;
    void addRenderTexture(RenderTextureBindingPoint bindingPoint, const RenderTexture& renderTexture);
    void addDepthTexture(const DepthTexture& depthTexture);
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
    DepthTexture mDepthTexture;
    Color mClearColor;
    float mClearDepthValue;
};


} // namespace Syrinx