#include "SyrinxRenderState.h"
#include <Common/SyrinxAssert.h>

namespace Syrinx {

ColorBlendState& ColorBlendState::setBlendEnable(uint32_t attachmentIndex, bool enable)
{
    resizeAttachmentStateSize(attachmentIndex + 1);
    mAttachmentBlendStateList[attachmentIndex].enableBlend = enable;
    return *this;
}


ColorBlendState& ColorBlendState::setColorBlendFunc(uint32_t attachmentIndex, const BlendFactor& srcBlendFactor, const BlendOp& blendOp, const BlendFactor& dstBlendFactor)
{
    resizeAttachmentStateSize(attachmentIndex + 1);

    auto& attachmentBlendState = mAttachmentBlendStateList[attachmentIndex];
    attachmentBlendState.srcColorBlendFactor = srcBlendFactor;
    attachmentBlendState.colorBlendOp = blendOp;
    attachmentBlendState.dstColorBlendFactor = dstBlendFactor;
    return *this;
}


ColorBlendState& ColorBlendState::setAlphaBlendFunc(uint32_t attachmentIndex, const BlendFactor& srcBlendFactor, const BlendOp& blendOp, const BlendFactor& dstBlendFactor)
{
    resizeAttachmentStateSize(attachmentIndex + 1);

    auto& attachmentBlendState = mAttachmentBlendStateList[attachmentIndex];
    attachmentBlendState.srcAlphaBlendFactor = srcBlendFactor;
    attachmentBlendState.alphaBlendOp = blendOp;
    attachmentBlendState.dstAlphaBlendFactor = dstBlendFactor;
    return *this;
}


const ColorBlendState::AttachmentBlendState& ColorBlendState::getAttachmentBlendState(uint32_t attachmentIndex) const
{
    if (attachmentIndex >= mAttachmentBlendStateList.size()) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
            "fail to get attachment blend state with index = [{}] because the index is greater than the size of attachments", attachmentIndex);
    }
    return mAttachmentBlendStateList[attachmentIndex];
}


const std::vector<ColorBlendState::AttachmentBlendState>& ColorBlendState::getAttachmentBlendStateList() const
{
    return mAttachmentBlendStateList;
}


void ColorBlendState::resizeAttachmentStateSize(uint32_t size)
{
    if (size > mAttachmentBlendStateList.size()) {
        mAttachmentBlendStateList.resize(size);
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