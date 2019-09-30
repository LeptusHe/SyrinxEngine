#include <GUI/SyrinxGui.h>
#include <Component/SyrinxController.h>
#include <Pipeline/SyrinxRenderPass.h>
#include <Pipeline/SyrinxScriptableRenderPipeline.h>

namespace Syrinx {

class EditorPipeline : public IScriptableRenderPipeline {
public:
    EditorPipeline();

    void onFrameRender(RenderContext& renderContext) override;
    void onGuiRender(Gui& gui) override;
    void addRenderPass(RenderPass *renderPass);
    void addCameraController(Controller *controller);

private:
    void importScene(const std::string& path);

private:
    bool mOpenFileDialog = false;

private:
    std::vector<RenderPass*> mRenderPassList;
    Controller *mCameraController;
};

} // namespace Syrinx