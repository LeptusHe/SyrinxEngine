#pragma once
#include <Graphics/SyrinxRenderContext.h>
#include <Scene/SyrinxScene.h>

namespace Syrinx {

class IScriptableRenderPipeline {
public:
    explicit IScriptableRenderPipeline(const std::string& name) : mName(name) {}

    virtual void onInit(Scene *scene) { };
    virtual void onFrameRender(RenderContext& renderContext) { };
    virtual void onGuiRender(Gui& gui) { };
    const std::string& getName() const { return mName; }

private:
    std::string mName;
};

} // namespace Syrinx