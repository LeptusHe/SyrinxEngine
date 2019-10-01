#include <GUI/SyrinxGui.h>
#include <Component/SyrinxController.h>
#include <Pipeline/SyrinxRenderPass.h>
#include <Pipeline/SyrinxScriptableRenderPipeline.h>
#include <Script/SyrinxLuaCommon.h>
#include "SyrinxLuaScriptRunner.h"

namespace Syrinx {

class EditorPipeline : public IScriptableRenderPipeline {
public:
    EditorPipeline();
    ~EditorPipeline() = default;
    void onFrameRender(RenderContext& renderContext) override;
    void onGuiRender(Gui& gui) override;
    void addCameraController(Controller *controller);

private:
    void importScene(const std::string& path);
    void importScript(const std::string& path);

private:
    bool mLoadScene = false;
    bool mLoadScript = false;

private:
    Controller *mCameraController;
    LuaScriptRunner mScriptRunner;
};

} // namespace Syrinx