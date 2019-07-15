#include "SyrinxRenderContext.h"
#include <Common/SyrinxAssert.h>
#include "SyrinxConstantTranslator.h"

namespace Syrinx {

void RenderContext::pushRenderState()
{
    mRenderStateStack.push_back(mRenderState);
}


void RenderContext::popRenderState()
{
    if (mRenderStateStack.empty()) {
        SYRINX_THROW_EXCEPTION(ExceptionCode::InvalidState, "fail to pop render state because the render state stack is empty");
    }

    mRenderState = mRenderStateStack.back();
    mRenderStateStack.pop_back();
}


void RenderContext::clearRenderTarget(RenderTarget *renderTarget, const Color& color)
{
    HardwareResource::ResourceHandle fboId = 0;
    if (renderTarget) {
        fboId = renderTarget->getHandle();
    }
    glClearNamedFramebufferfv(fboId, GL_COLOR, 0, color);
}


void RenderContext::setRenderState(const RenderState *renderState)
{
    mRenderState = renderState;
}


void RenderContext::drawIndexed(uint32_t indexCount)
{
    drawIndexed(indexCount, 0);
}


void RenderContext::drawIndexed(uint32_t indexCount, uint32_t indexOffset)
{
    auto vertexInputState = mRenderState->getVertexInputState();
    glBindVertexArray(vertexInputState->getHandle());

    auto indexType = vertexInputState->getIndexBuffer()->getIndexType();
    auto numIndex = vertexInputState->getIndexBuffer()->getNumIndexes();
    if (indexCount > numIndex) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                   "index count [{}] is greater thant the index number of vertex input state [{}]",
                                   indexCount, numIndex);
    }

    auto glIndexType = GL_UNSIGNED_INT;
    if (indexType._value == IndexType::UINT16) {
        glIndexType = GL_UNSIGNED_SHORT;
    }
    glDrawElements(GL_TRIANGLES, indexCount, glIndexType, reinterpret_cast<const void*>(indexOffset));
}


bool RenderContext::isValidToDraw() const
{
    return mRenderState;
}


void RenderContext::prepareDraw()
{
    if (!isValidToDraw()) {
        SYRINX_THROW_EXCEPTION(ExceptionCode::InvalidState, "fail to draw indexed");
    }

    auto viewportOffsetX = mRenderState->viewportState.viewport.offset.x;
    auto viewportOffsetY = mRenderState->viewportState.viewport.offset.y;
    auto viewportExtentX = mRenderState->viewportState.viewport.extent.x;
    auto viewportExtentY = mRenderState->viewportState.viewport.extent.y;
    glViewport(viewportOffsetX, viewportOffsetY, viewportExtentX, viewportExtentY);

    auto programPipeline = mRenderState->getProgramPipeline();
    auto renderTarget = mRenderState->getRenderTarget();

    if (!renderTarget) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    } else {
        glBindFramebuffer(GL_FRAMEBUFFER, renderTarget->getHandle());
    }

    setDepthState();
    setCullState();
    setColorBlendState();

    glBindProgramPipeline(programPipeline->getHandle());
}


void RenderContext::setColorBlendState() const
{
    SYRINX_EXPECT(mRenderState);
    auto& colorBlendState = mRenderState->colorBlendState;
    auto& attachmentBlendStateList = colorBlendState.getAttachmentBlendStateList();
    auto renderTarget = mRenderState->getRenderTarget();

    bool disableBlend = true;
    for (const auto& attachmentBlendState : attachmentBlendStateList) {
        if (attachmentBlendState.enableBlend) {
            glEnable(GL_BLEND);
            disableBlend = false;
        }
    }
    if (disableBlend) {
        glDisable(GL_BLEND);
        return;
    }

    GLuint attachmentIndex = 0;
    for (int i = 0; i < attachmentBlendStateList.size(); ++ i) {
        auto& attachmentBlendState = attachmentBlendStateList[i];

        if (renderTarget) {
            if (renderTarget->getColorAttachment(i)) {
                attachmentIndex = i;
            } else {
                SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                    "fail to set blend state because there is no color attachment [binding point = {}] in render target [{}]",
                    i, renderTarget->getName());
            }
        }

        auto colorBlendOp = ConstantTranslator::getBlendOp(attachmentBlendState.colorBlendOp);
        auto alphaBlendOp = ConstantTranslator::getBlendOp(attachmentBlendState.alphaBlendOp);
        glBlendEquationSeparatei(attachmentIndex, colorBlendOp, alphaBlendOp);

        auto colorSrcBlendFactor = ConstantTranslator::getBlendFactor(attachmentBlendState.srcColorBlendFactor);
        auto colorDstBlendFactor = ConstantTranslator::getBlendFactor(attachmentBlendState.dstColorBlendFactor);
        auto alphaSrcBlendFactor = ConstantTranslator::getBlendFactor(attachmentBlendState.srcAlphaBlendFactor);
        auto alphaDstBlendFactor = ConstantTranslator::getBlendFactor(attachmentBlendState.dstAlphaBlendFactor);
        glBlendFuncSeparatei(attachmentIndex, colorSrcBlendFactor, colorDstBlendFactor, alphaSrcBlendFactor, alphaDstBlendFactor);
    }
}


void RenderContext::setCullState() const
{
    SYRINX_EXPECT(mRenderState);

    auto& cullMode = mRenderState->rasterizationState.cullMode;
    if (cullMode._value == CullMode::None) {
        glDisable(GL_CULL_FACE);
    } else {
        glEnable(GL_CULL_FACE);
        if (cullMode._value == CullMode::Front) {
            glCullFace(GL_FRONT);
        } else if (cullMode._value == CullMode::Back) {
            glCullFace(GL_BACK);
        } else {
            glCullFace(GL_FRONT_AND_BACK);
        }
    }
}


void RenderContext::setDepthState() const
{
    SYRINX_EXPECT(mRenderState);

    if (mRenderState->depthStencilState.enableDepthTest) {
        glEnable(GL_DEPTH_TEST);
    } else {
        glDisable(GL_DEPTH_TEST);
    }
}

} // namespace Syrinx
