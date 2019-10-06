#include "SyrinxEditorPipeline.h"
#include <Widgets/SyrinxFileDialog.h>
#include <Pipeline/SyrinxEngine.h>
#include <Script/SyrinxLuaBinder.h>

namespace Syrinx {

EditorPipeline::EditorPipeline()
    : IScriptableRenderPipeline("SyrinxEditorPipeline")
    , mCameraController(nullptr)
{
    SYRINX_EXPECT(!mCameraController);
}


void EditorPipeline::onFrameRender(RenderContext& renderContext)
{
    renderContext.clearRenderTarget(nullptr, Color(0.0, 0.0, 0.0, 1.0));
    renderContext.clearDepth(nullptr, 1.0);

    auto frameRender = mScriptRunner.get<sol::function>("render");
    if (frameRender) {
        try {
            auto cameraList = getCameraList();
            frameRender(renderContext, cameraList, getActiveScene());
        } catch (std::exception& e) {
            mScriptRunner.set("render", nullptr);
            SYRINX_ERROR_FMT("fail to call function [render] in script file: error [{}]", e.what());
        }
    }
}


void EditorPipeline::onGuiRender(Gui& gui)
{
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Load Scene")) {
                mLoadScene = true;
            }
            if (ImGui::MenuItem("Load Script")) {
                mLoadScript = true;
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    auto windowSize = getWindowSize();
    float width = windowSize.x;
    float height = windowSize.y;
/*
    ImGuiWindowFlags windowFlags =
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_MenuBar |
        ImGuiWindowFlags_NoCollapse;
    ImGui::Begin("Syrinx Editor", nullptr, windowFlags);
    ImGui::SetWindowPos(ImVec2(0, 0));
    ImGui::SetWindowSize(ImVec2(width, height));

    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("Window")) {
            ImGui::MenuItem("Open Scene", nullptr);
            ImGui::EndMenu();
        }

        ImGui::EndMenuBar();
    }
    ImGui::End();
*/

    if (mLoadScene) {
        auto& fileDialog = FileDialog::getInstance();
        auto[isSelected, path] = fileDialog.open("file dialog", width / 2, height / 2);
        if (isSelected) {
            importScene(path);
            mLoadScene = false;
        }
    }

    if (mLoadScript) {
        auto& fileDialog = FileDialog::getInstance();
        auto [isSelected, path] = fileDialog.open("file dialog", width / 2, height / 2);
        if (isSelected) {
            importScript(path);
            mLoadScript = false;
        }
    }
}


void EditorPipeline::addCameraController(Controller *controller)
{
    mCameraController = controller;
}


void EditorPipeline::importScene(const std::string& path)
{
    auto engine = getEngine();
    SYRINX_ASSERT(engine);

    try {
        auto sceneManager = engine->getSceneManager();
        auto scene = sceneManager->importScene(path);
        engine->setActiveScene(scene);
        const auto& cameraEntityList = scene->getEntitiesWithComponent<Camera>();
        for (auto cameraEntity : cameraEntityList) {
            cameraEntity->addController(mCameraController);
        }
    } catch (std::exception& e) {
        SYRINX_ERROR_FMT("fail to import scene because [{}]", e.what());
        return;
    }

}


void EditorPipeline::importScript(const std::string& path)
{
    auto engine = getEngine();
    SYRINX_ASSERT(engine);
    auto fileManager = engine->getFileManager();
    SYRINX_ASSERT(fileManager);

    auto [fileExist, filePath] = fileManager->findFile(path);
    if (!fileExist) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileNotFound, "fail to load script file [{}]", path);
    }

    auto fileStream = fileManager->openFile(filePath, FileAccessMode::READ);
    if (!fileStream) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileSystemError, "fail to open file [{}]", filePath);
    }
    auto source = fileStream->getAsString();
    mScriptRunner.run(source);
}

} // namespace Syrinx
