#pragma once
#include "SyrinxRenderState.h"

namespace Syrinx {

class RenderContext {
public:
    RenderContext() = default;
    void pushRenderState();
    void popRenderState();
    void clearRenderTarget(RenderTarget *renderTarget, const Color& color);
    void clearDepth(RenderTarget *renderTarget, float depth);
    void setRenderState(RenderState *renderState);
    void setColorBlendState() const;
    void setCullState() const;
    void setDepthState() const;
    void prepareDraw();
    void drawIndexed(uint32_t indexCount);
    void drawIndexed(uint32_t indexCount, uint32_t indexOffset);
    RenderState *getRenderState();

protected:
    bool isValidToDraw() const;

private:

private:
    RenderState *mRenderState = nullptr;
    std::vector<RenderState*> mRenderStateStack;
};

} // namespace Syrinx