#pragma once
#include <Graphics/SyrinxRenderContext.h>
#include <Scene/SyrinxScene.h>

namespace Syrinx {

class IScriptableRenderPipeline {
public:
    explicit IScriptableRenderPipeline(const std::string& name) : mName(name) {}

    virtual void onInit(Scene& scene) = 0;
    virtual void onFrameRender(RenderContext& renderContext) = 0;
    const std::string& getName() const { return mName; }

private:
    std::string mName;
};

} // namespace Syrinx