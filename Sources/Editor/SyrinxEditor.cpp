#include "SyrinxEditor.h"
#include <Exception/SyrinxException.h>

namespace Syrinx {

void Editor::init(int width, int height)
{
    SYRINX_EXPECT(!mEngine);

    if (width <= 0 || height <= 0) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
            "fail to initialize editor because the width or height is invalid", width, height);
    }

    mEngine = std::make_unique<Engine>();
    mEngine->init();

    auto fileManager = mEngine->getFileManager();
    fileManager->addSearchPath("Medias/Library/");

    auto shaderManager = mEngine->getShaderManager();
    shaderManager->addShaderSearchPath("Medias/Library/");
    shaderManager->addShaderSearchPath("Medias/Library/Shader/Unlit");

    mEngine->createWindow("SyrinxEditor", width, height);

    mRenderPipeline = std::make_unique<EditorPipeline>();
    mEngine->addRenderPipeline(mRenderPipeline.get());
    mEngine->setActiveRenderPipeline(mRenderPipeline.get());
}


void Editor::run()
{
    mEngine->run();
}

} // namespace Syrinx