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


Scene* IScriptableRenderPipeline::getActiveScene() const
{
    auto engine = getEngine();
    SYRINX_ASSERT(engine);
    return engine->getActiveScene();
}


std::vector<Camera*> IScriptableRenderPipeline::getCameraList() const
{
    auto scene = getActiveScene();
    if (!scene) {
        return {};
    }

    std::vector<Camera*> cameraList;
    auto cameraEntityList = scene->getEntitiesWithComponent<Camera>();
    for (auto cameraEntity : cameraEntityList) {
        SYRINX_ASSERT(cameraEntity->hasComponent<Camera>());
        auto& camera = cameraEntity->getComponent<Camera>();
        cameraList.push_back(&camera);
    }
    return cameraList;
}


Engine* IScriptableRenderPipeline::getEngine() const
{
    return mEngine;
}

} // namespace Syrinx
