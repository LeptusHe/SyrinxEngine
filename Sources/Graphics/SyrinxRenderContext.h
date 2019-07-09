#pragma once
#include "SyrinxRenderState.h"

namespace Syrinx {

class RenderContext {
public:
    RenderContext() = default;
    void pushRenderState();
    void popRenderState();
    void clearRenderTarget(RenderTarget *renderTarget, const Color& color);
    void setRenderState(const RenderState *renderState);
    void prepareDraw();
    void drawIndexed(uint32_t indexCount);
    void drawIndexed(uint32_t indexCount, uint32_t indexOffset);

protected:
    bool isValidToDraw() const;

private:
    const RenderState *mRenderState = nullptr;
    std::vector<const RenderState*> mRenderStateStack;
};

} // namespace Syrinx