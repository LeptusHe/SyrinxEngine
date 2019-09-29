#include "SyrinxEditorPipeline.h"
#include <Widgets/SyrinxFileDialog.h>
#include <Pipeline/SyrinxEngine.h>
#include "SyrinxLightingPass.h"

namespace Syrinx {

EditorPipeline::EditorPipeline() : IScriptableRenderPipeline("SyrinxEditorPipeline")
{

}


void EditorPipeline::onFrameRender(RenderContext& renderContext)
{
    renderContext.clearRenderTarget(nullptr, Color(0.0, 0.0, 0.0, 1.0));
    renderContext.clearDepth(nullptr, 1.0);

    for (auto renderPass : mRenderPassList) {
        renderPass->onFrameRender(renderContext);
    }
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

    auto windowSize = getWindowSize();

    ImGuiWindowFlags windowFlags =
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_MenuBar |
        ImGuiWindowFlags_NoCollapse;

    ImGui::Begin("Syrinx Editor", nullptr, windowFlags);
    ImGui::SetWindowPos(ImVec2(0, 0));

    float width = windowSize.x;
    float height = windowSize.y;
    ImGui::SetWindowSize(ImVec2(width, height));

    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("Window")) {
            ImGui::MenuItem("Open Scene", nullptr);
            ImGui::EndMenu();
        }

        ImGui::EndMenuBar();
    }
    ImGui::End();

    auto& fileDialog = FileDialog::getInstance();
    auto [isSelected, path] = fileDialog.open("file dialog", width / 2, height / 2);
    if (isSelected) {
        importScene(path);
    }

    for (auto renderPass : mRenderPassList) {
        renderPass->onGuiRender(gui);
    }
}


void EditorPipeline::addRenderPass(RenderPass *renderPass)
{
    SYRINX_EXPECT(renderPass);
    mRenderPassList.push_back(renderPass);
}


void EditorPipeline::importScene(const std::string& path)
{
    auto engine = getEngine();
    SYRINX_ASSERT(engine);

    auto sceneManager = engine->getSceneManager();
    auto scene =sceneManager->importScene(path);
    engine->setActiveScene(scene);

    Entity* cameraEntity = nullptr;
    const auto& entityList = scene->getEntityList();
    for (const auto& entity : entityList) {
        if (entity->hasComponent<Camera>()) {
            cameraEntity = entity;
            break;
        }
    }

    const auto& meshEntityList = scene->getEntitiesWithComponent<Renderer>();
    for (auto renderPass : mRenderPassList) {
        renderPass->setCamera(cameraEntity);
        renderPass->addEntityList(meshEntityList);
    }
}

} // namespace Syrinx
