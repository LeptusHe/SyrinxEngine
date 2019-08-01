#include <GUI/SyrinxGui.h>
#include <Pipeline/SyrinxScriptableRenderPipeline.h>

namespace Syrinx {

class EditorPipeline : public IScriptableRenderPipeline {
public:
    EditorPipeline();

    void onFrameRender(RenderContext& renderContext) override;
    void onGuiRender(Gui& gui) override;
};

} // namespace Syrinx