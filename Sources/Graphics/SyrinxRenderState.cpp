#include "SyrinxRenderState.h"
#include <Common/SyrinxAssert.h>

namespace Syrinx {

ColorBlendState& ColorBlendState::setBlendEnable(uint32_t attachmentIndex, bool enable)
{
    resizeAttachmentStateSize(attachmentIndex + 1);
    mAttachmentBlendState[attachmentIndex].enableBlend = enable;
    return *this;
}


ColorBlendState& ColorBlendState::setColorBlendFunc(uint32_t attachmentIndex, const BlendFactor& srcBlendFactor, const BlendOp& blendOp, const BlendFactor& dstBlendFactor)
{
    resizeAttachmentStateSize(attachmentIndex + 1);

    auto& attachmentBlendState = mAttachmentBlendState[attachmentIndex];
    attachmentBlendState.srcColorBlendFactor = srcBlendFactor;
    attachmentBlendState.colorBlendOp = blendOp;
    attachmentBlendState.dstColorBlendFactor = dstBlendFactor;
    return *this;
}


ColorBlendState& ColorBlendState::setAlphaBlendFunc(uint32_t attachmentIndex, const BlendFactor& srcBlendFactor, const BlendOp& blendOp, const BlendFactor& dstBlendFactor)
{
    resizeAttachmentStateSize(attachmentIndex + 1);

    auto& attachmentBlendState = mAttachmentBlendState[attachmentIndex];
    attachmentBlendState.srcAlphaBlendFactor = srcBlendFactor;
    attachmentBlendState.alphaBlendOp = blendOp;
    attachmentBlendState.dstAlphaBlendFactor = dstBlendFactor;
    return *this;
}


void ColorBlendState::resizeAttachmentStateSize(uint32_t size)
{
    if (size > mAttachmentBlendState.size()) {
        mAttachmentBlendState.resize(size);
    }
}




void RenderState::setVertexInputState(const VertexInputState *vertexInputState)
{
    SYRINX_EXPECT(vertexInputState && vertexInputState->isCreated());
    mVertexInputState = vertexInputState;
    SYRINX_ENSURE(mVertexInputState);
}


void RenderState::setProgramPipeline(const ProgramPipeline *programPipeline)
{
    SYRINX_EXPECT(programPipeline && programPipeline->isCreated());
    mProgramPipeline = programPipeline;
    SYRINX_ENSURE(mProgramPipeline);
}


void RenderState::setRenderTarget(const RenderTarget *renderTarget)
{
    mRenderTarget = renderTarget;
}


const VertexInputState* RenderState::getVertexInputState() const
{
    return mVertexInputState;
}


const ProgramPipeline* RenderState::getProgramPipeline() const
{
    return mProgramPipeline;
}


const RenderTarget* RenderState::getRenderTarget() const
{
    return mRenderTarget;
}

} // namespace Syrinx