#include "SyrinxScriptableRenderPipeline.h"
#include <Pipeline/SyrinxEngine.h>

namespace Syrinx {

void IScriptableRenderPipeline::setEngine(Engine *engine)
{
    SYRINX_EXPECT(engine);
    mEngine = engine;
}


Vector2i IScriptableRenderPipeline::getWindowSize() const
{
    auto renderWindow = mEngine->getWindow();
    SYRINX_ASSERT(renderWindow);

    uint32_t width = renderWindow->getWidth();
    uint32_t height = renderWindow->getHeight();

    return Vector2i(width, height);
}


Engine* IScriptableRenderPipeline::getEngine() const
{
    return mEngine;
}

} // namespace Syrinx
