#pragma once
#include <memory>
#include <Component/SyrinxCamera.h>
#include <Component/SyrinxController.h>
#include <Pipeline/SyrinxEngine.h>
#include <Pipeline/SyrinxRenderPass.h>
#include "SyrinxEditorPipeline.h"

namespace Syrinx {

class Editor {
public:
    void init(unsigned int width, unsigned int height);
    void run();

private:
    std::unique_ptr<Engine> mEngine;
    std::unique_ptr<EditorPipeline> mRenderPipeline;
    std::unique_ptr<Camera> mCamera;
    std::unique_ptr<RenderPass> mRenderPass;
    std::unique_ptr<RenderState> mRenderState;
    std::unique_ptr<Controller> mCameraController;
};

} // namespace Syrinx