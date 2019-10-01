#pragma once
#include <Graphics/SyrinxRenderContext.h>
#include <Scene/SyrinxScene.h>
#include <GUI/SyrinxGui.h>

namespace Syrinx {

class Engine;


class IScriptableRenderPipeline {
public:
    explicit IScriptableRenderPipeline(const std::string& name) : mName(name) {}

    virtual void onInit(Scene *scene) { };
    virtual void onFrameRender(RenderContext& renderContext) { };
    virtual void onGuiRender(Gui& gui) { };
    const std::string& getName() const { return mName; }
    void setEngine(Engine *engine);
    Vector2i getWindowSize() const;
    Scene* getActiveScene() const;
    std::vector<Camera*> getCameraList() const;

protected:
    Engine* getEngine() const;

private:
    Engine *mEngine = nullptr;
    std::string mName;
};

} // namespace Syrinx