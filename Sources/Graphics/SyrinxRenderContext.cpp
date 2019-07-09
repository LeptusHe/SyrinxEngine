#include "SyrinxRenderContext.h"
#include <Common/SyrinxAssert.h>
#include "HardwareResource/SyrinxConstantTranslator.h"

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

    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto viewportOffsetX = mRenderState->viewportState.viewport.offset.x;
    auto viewportOffsetY = mRenderState->viewportState.viewport.offset.y;
    auto viewportExtentX = mRenderState->viewportState.viewport.extent.x;
    auto viewportExtentY = mRenderState->viewportState.viewport.extent.y;
    glViewport(viewportOffsetX, viewportOffsetY, viewportExtentX, viewportExtentY);

    if (mRenderState->depthStencilState.enableDepthTest) {
        glEnable(GL_DEPTH_TEST);
    } else {
        glDisable(GL_DEPTH_TEST);
    }
    glDisable(GL_CULL_FACE);

    auto programPipeline = mRenderState->getProgramPipeline();
    auto renderTarget = mRenderState->getRenderTarget();

    if (!renderTarget) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    } else {
        glBindFramebuffer(GL_FRAMEBUFFER, renderTarget->getHandle());
    }
    glBindProgramPipeline(programPipeline->getHandle());
}

} // namespace Syrinx
