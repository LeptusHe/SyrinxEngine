#include "SyrinxEditor.h"
#include <Exception/SyrinxException.h>
#include "SyrinxLightingPass.h"
#include "SyrinxCameraController.h"

namespace Syrinx {

void Editor::init(unsigned int width, unsigned int height)
{
    SYRINX_EXPECT(!mEngine);

    if (width <= 0 || height <= 0) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
            "fail to initialize editor because the width or height is invalid", width, height);
    }

    mEngine = std::make_unique<Engine>();
    mEngine->init();

    auto fileManager = mEngine->getFileManager();
    auto fileSystem = fileManager->getFileSystem();
    fileManager->addSearchPath(fileSystem->getWorkingDirectory());
    fileManager->addSearchPath("Medias/Library/");

    auto shaderManager = mEngine->getShaderManager();
    shaderManager->addShaderSearchPath("Medias/Library/");
    shaderManager->addShaderSearchPath("Medias/Library/Shader/Unlit");

    mEngine->createWindow("SyrinxEditor", width, height);

    mRenderPipeline = std::make_unique<EditorPipeline>();
    mEngine->addRenderPipeline(mRenderPipeline.get());
    mEngine->setActiveRenderPipeline(mRenderPipeline.get());

    mCamera = std::make_unique<Camera>("main camera");
    mCamera->setViewportRect({0, 0, width, height});

    mRenderPass = std::make_unique<LightingPass>("lighting");
    mRenderPass->setShaderName("display-world-normal.shader");

    mRenderState = std::make_unique<RenderState>();
    mRenderState->viewportState.viewport.extent = {width, height};
    mRenderPass->setRenderState(mRenderState.get());

    mRenderPipeline->addRenderPass(mRenderPass.get());

    mCameraController = std::make_unique<CameraMotionController>();
    mRenderPipeline->addCameraController(mCameraController.get());
}


void Editor::run()
{
    mEngine->run();
}

} // namespace Syrinx