#pragma once
#include <memory>
#include <Pipeline/SyrinxEngine.h>
#include "SyrinxEditorPipeline.h"

namespace Syrinx {

class Editor {
public:
    void init(int width, int height);
    void run();

private:
    std::unique_ptr<Engine> mEngine;
    std::unique_ptr<EditorPipeline> mRenderPipeline;
};

} // namespace Syrinx