#include "SyrinxEditorPipeline.h"
#include <Widgets/SyrinxFileDialog.h>

namespace Syrinx {

EditorPipeline::EditorPipeline() : IScriptableRenderPipeline("SyrinxEditorPipeline")
{

}


void EditorPipeline::onFrameRender(RenderContext& renderContext)
{
    renderContext.clearRenderTarget(nullptr, Color(0.0, 0.0, 0.0, 1.0));
    renderContext.clearDepth(nullptr, 1.0);
}


void EditorPipeline::onGuiRender(Gui& gui)
{
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Load Scene")) {}
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    ImVec4 color = ImVec4(1.0, 0.0, 0.0, 1.0);
    ImGui::TextColored(color, "Colored Text");
    ImGui::Text("Syrinx GUI Render Sample");
    static bool show = false;

    ImGui::Checkbox("Demo Window", &show);

    auto& fileDialog = FileDialog::getInstance();
    fileDialog.open("file dialog");
}

} // namespace Syrinx
