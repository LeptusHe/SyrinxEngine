#include <Logging/SyrinxLogManager.h>
#include <Pipeline/SyrinxEngine.h>
#include <Pipeline/SyrinxDisplayDevice.h>
#include <GUI/SyrinxGui.h>
#include <Eigen/Eigen/Eigen>

Eigen::VectorXf Interpolation(std::vector<Eigen::Vector2f> points)
{
    int n = points.size();

    // Ax=y
    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(n, n);
    Eigen::VectorXf y = Eigen::VectorXf::Zero(n);

    for (int i = 0; i < n; ++ i) {
        const auto& p = points[i];
        for (int j = 0; j < n; ++ j) {
            A(i, j) = std::powf(p.x(), static_cast<float>(j));
        }
        y(i) = p.y();
    }
    return A.inverse() * y;
}


float Poly(float* factors, int num, float x)
{
    float result = 0;
    for (int i = 0; i < num; ++ i) {
        result += factors[i] * std::powf(x, i);
    }
    return result;
}


void DrawPoints(const std::vector<Eigen::Vector2f>& points)
{
    float *x = new float[points.size()];
    float *y = new float[points.size()];

    for (int i = 0; i < points.size(); ++ i) {
        x[i] = points[i].x();
        y[i] = points[i].y();
    }

    ImPlot::PlotScatter("points", x, y, points.size());
}



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

    std::vector<Eigen::Vector2f> points;

    bool solved = false;
    Eigen::VectorXf result;

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

        if (ImPlot::BeginPlot("Plot")) {
            bool hovered = ImPlot::IsPlotHovered();
            bool mouseClicked = ImGui::IsMouseDoubleClicked(0);
            bool keyPressed = ImGui::GetIO().KeyCtrl;

            if (mouseClicked) {
                SYRINX_INFO("mouse pressed");
            }

            if (keyPressed) {
                SYRINX_INFO("key pressed");
            }

            if (ImPlot::IsPlotHovered() && ImGui::IsMouseDoubleClicked(0)) {
                ImPlotPoint p = ImPlot::GetPlotMousePos();
                points.push_back(Eigen::Vector2f(p.x, p.y));
                SYRINX_INFO_FMT("x= {}, y={}", p.x, p.y);
            }
            DrawPoints(points);

            if (solved) {
                int count = 20;

                float *x = new float[count];
                float *y = new float[count];
                for (int i = 0; i < count; ++ i) {
                    x[i] = 0 + i * 1 / static_cast<float>(count);
                    y[i] = Poly(result.data(), result.size(), x[i]);
                }

                ImPlot::PlotLine("y", x, y, count);
            }
            ImPlot::EndPlot();
        }

        if (ImGui::Button("Interploate")) {
            result = Interpolation(points);
            solved = true;
        }

        if (ImGui::Button("Clear")) {
            solved = false;
            points.clear();
        }

        SYRINX_ASSERT(activeFont);
        ImGui::Checkbox("Demo Window", &show);
        ImGui::PopFont();
        ImGui::End();

        gui.render(&renderContext);
        renderWindow->swapBuffer();
    }

    return 0;
}