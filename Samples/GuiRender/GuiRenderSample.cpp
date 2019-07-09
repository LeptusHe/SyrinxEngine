#include <Logging/SyrinxLogManager.h>
#include <Pipeline/SyrinxEngine.h>
#include <Pipeline/SyrinxDisplayDevice.h>
#include <GUI/SyrinxGui.h>

int main(int argc, char *argv[])
{
    auto logManager = std::make_unique<Syrinx::LogManager>();
    Syrinx::DisplayDevice displayDevice;

    displayDevice.setMajorVersionNumber(4);
    displayDevice.setMinorVersionNumber(6);
    displayDevice.setDebugMessageHandler(Syrinx::DefaultDebugHandler);
    auto renderWindow = displayDevice.createWindow("GUI Render Sample", 800, 600);

    Syrinx::FileManager fileManager;
    Syrinx::HardwareResourceManager hardwareResourceManager;
    Syrinx::ShaderManager shaderManager(&fileManager, &hardwareResourceManager);

    fileManager.addSearchPath(".");
    fileManager.addSearchPath("../../Medias");
    shaderManager.addShaderSearchPath("../../Medias/Library");

    Syrinx::Gui gui(&fileManager, &shaderManager, &hardwareResourceManager);
    gui.init();
    gui.onWindowResize(renderWindow->getWidth(), renderWindow->getHeight());

    Syrinx::Input input(renderWindow->fetchWindowHandle());

    Syrinx::RenderContext renderContext;
    gui.addFont("SourceCodePro-Black", "SourceCodePro-Black.ttf");
    while (renderWindow->isOpen()) {
        gui.beginFrame();
        glfwPollEvents();
        gui.onInputEvents(&input);

        gui.setActiveFont("SourceCodePro-Black");

        renderContext.clearRenderTarget(nullptr, Syrinx::Color(0.0, 0.5, 0.5, 1.0));
        auto activeFont = gui.getActiveFont();
        ImGui::PushFont(activeFont);
        ImGui::Begin("Text");
        ImVec4 color = ImVec4(1.0, 0.0, 0.0, 1.0);
        ImGui::TextColored(color, "Colored Text");
        ImGui::Text("Syrinx GUI Render Sample");
        static bool show = false;

        SYRINX_ASSERT(activeFont);
        ImGui::Checkbox("Demo Window", &show);
        ImGui::PopFont();
        ImGui::End();
        gui.render(&renderContext);
        renderWindow->swapBuffer();
    }

    return 0;
}