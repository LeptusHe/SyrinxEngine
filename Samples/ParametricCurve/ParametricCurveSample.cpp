#include <Logging/SyrinxLogManager.h>
#include <Pipeline/SyrinxEngine.h>

#include "PolynomialInterpolation.h"
#include "GaussianInterpolation.h"
#include "PolynomialFitting.h"
#include "ParametricCurve.h"


void DrawPoints(const std::vector<Eigen::Vector2d>& points)
{
    auto *x = new double[points.size()];
    auto *y = new double[points.size()];

    for (int i = 0; i < points.size(); ++ i) {
        x[i] = points[i].x();
        y[i] = points[i].y();
    }

    ImPlot::PlotScatter("points", x, y, points.size());
}


int main(int argc, char *argv[])
{
    const int width = 1600;
    const int height = 800;

    auto logManager = std::make_unique<Syrinx::LogManager>();
    Syrinx::DisplayDevice displayDevice;

    displayDevice.setMajorVersionNumber(4);
    displayDevice.setMinorVersionNumber(6);
    displayDevice.setDebugMessageHandler(Syrinx::DefaultDebugHandler);
    auto renderWindow = displayDevice.createWindow("GUI Render Sample", width, height);

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

    std::vector<Eigen::Vector2d> points;

    bool solved = false;
    Eigen::VectorXd result;

    float lambda = 1.0f;
    ParametrizationMethod method = ParametrizationMethod::Chordal;
    const char* methodName[] = {"Uniform", "Chordal"};

    ParametricCurve parametricCurve(0.0f);

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

        ImGui::Begin("Windows");

        int item = static_cast<int>(method);
        if (ImGui::Combo("Parametrization Method", &item, methodName, static_cast<int>(ParametrizationMethod::Count))) {
            method = static_cast<ParametrizationMethod>(item);
            if (solved) {
                parametricCurve.Solve(points, method);
            }
        }

        if (ImGui::SliderFloat("Lambda", &lambda, 0.0, 1.0)) {
            parametricCurve = ParametricCurve(lambda);
            if (solved) {
                parametricCurve.Solve(points, method);
            }
        }
        ImGui::Spacing();

        static bool show = false;
        const auto axisFlags = ImPlotAxisFlags_Lock;
        if (ImPlot::BeginPlot("Plot", nullptr, nullptr, ImVec2(width / 2.0f, height / 2.0f), ImPlotFlags_None, axisFlags, axisFlags)) {
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
                points.emplace_back(p.x, p.y);
                SYRINX_INFO_FMT("x= {}, y={}", p.x, p.y);
            }
            DrawPoints(points);

            if (solved) {
                int count = 50;

                auto x = new double[count];
                auto y = new double[count];
                for (int i = 0; i <= count; ++ i) {
                    double t = 0 + i * 1 / static_cast<double>(count);
                    auto pos = parametricCurve.F(t);
                    x[i] = pos.x();
                    y[i] = pos.y();
                }
                ImPlot::PlotLine("parametric curve", x, y, count);
            }
            ImPlot::EndPlot();
        }

        if (ImGui::Button("Calculation")) {
            parametricCurve.Solve(points, ParametrizationMethod::Chordal);
            solved = true;
        }

        ImGui::Spacing();
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