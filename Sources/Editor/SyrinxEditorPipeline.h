#include <GUI/SyrinxGui.h>
#include <Pipeline/SyrinxRenderPass.h>
#include <Pipeline/SyrinxScriptableRenderPipeline.h>

namespace Syrinx {

class EditorPipeline : public IScriptableRenderPipeline {
public:
    EditorPipeline();

    void onFrameRender(RenderContext& renderContext) override;
    void onGuiRender(Gui& gui) override;
    void addRenderPass(RenderPass *renderPass);

private:
    void importScene(const std::string& path);

private:
    std::vector<RenderPass*> mRenderPassList;
};

} // namespace Syrinx