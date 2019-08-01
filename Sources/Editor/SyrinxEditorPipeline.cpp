#include "SyrinxEditorPipeline.h"

namespace Syrinx {

EditorPipeline::EditorPipeline() : IScriptableRenderPipeline("SyrinxEditorPipeline")
{

}


void EditorPipeline::onFrameRender(RenderContext& renderContext)
{
    renderContext.clearRenderTarget(nullptr, Color(1.0, 0.0, 1.0, 1.0));
    renderContext.clearDepth(nullptr, 1.0);
}


void EditorPipeline::onGuiRender(Gui& gui)
{
    ImGui::Begin("Text");
    ImVec4 color = ImVec4(1.0, 0.0, 0.0, 1.0);
    ImGui::TextColored(color, "Colored Text");
    ImGui::Text("Syrinx GUI Render Sample");
    static bool show = false;

    ImGui::Checkbox("Demo Window", &show);
    ImGui::End();
}

} // namespace Syrinx
