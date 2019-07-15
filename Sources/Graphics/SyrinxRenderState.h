#pragma once
#include <vector>
#include <Math/SyrinxGeometry.h>
#include "HardwareResource/SyrinxVertexInputState.h"
#include "HardwareResource/SyrinxProgramPipeline.h"
#include "HardwareResource/SyrinxRenderTarget.h"

namespace Syrinx {

struct InputAssemblyState {
    PrimitiveTopology topology = PrimitiveTopology::Triangle;
};


struct ViewportState {
    Rect2D<uint32_t> viewport;
    bool enableScissor = false;
    Rect2D<uint32_t> scissor;
};


struct RasterizationState {
    PolygonMode polygonMode = PolygonMode::Fill;
    CullMode cullMode = CullMode::Back;
};


struct DepthStencilState {
    bool enableDepthTest = true;
    bool enableDepthWrite = true;
};


class ColorBlendState {
public:
    struct AttachmentBlendState {
        bool enableBlend = false;
        BlendFactor srcColorBlendFactor = BlendFactor::One;
        BlendFactor dstColorBlendFactor = BlendFactor::Zero;
        BlendOp colorBlendOp = BlendOp::Add;
        BlendFactor srcAlphaBlendFactor = BlendFactor::One;
        BlendFactor dstAlphaBlendFactor = BlendFactor::Zero;
        BlendOp alphaBlendOp = BlendOp::Add;
    };

    ColorBlendState& setBlendEnable(uint32_t attachmentIndex, bool enable);
    ColorBlendState& setColorBlendFunc(uint32_t attachmentIndex, const BlendFactor& srcBlendFactor, const BlendOp& blendOp, const BlendFactor& dstBlendFactor);
    ColorBlendState& setAlphaBlendFunc(uint32_t attachmentIndex, const BlendFactor& srcBlendFactor, const BlendOp& blendOp, const BlendFactor& dstBlendFactor);
    const AttachmentBlendState& getAttachmentBlendState(uint32_t attachmentIndex) const;
    const std::vector<AttachmentBlendState>& getAttachmentBlendStateList() const;

private:
    void resizeAttachmentStateSize(uint32_t size);

private:
    std::vector<AttachmentBlendState> mAttachmentBlendStateList{4};
};


class RenderState {
public:
    RenderState() = default;
    void setVertexInputState(const VertexInputState *vertexInputState);
    void setProgramPipeline(const ProgramPipeline *programPipeline);
    void setRenderTarget(const RenderTarget *renderTarget);
    const VertexInputState* getVertexInputState() const;
    const ProgramPipeline* getProgramPipeline() const;
    const RenderTarget* getRenderTarget() const;

public:
    InputAssemblyState inputAssemblyState;
    ViewportState viewportState;
    RasterizationState rasterizationState;
    DepthStencilState depthStencilState;
    ColorBlendState colorBlendState;

private:
    const VertexInputState *mVertexInputState = nullptr;
    const ProgramPipeline *mProgramPipeline = nullptr;
    const RenderTarget *mRenderTarget = nullptr;
};

} // namespace Syrinx