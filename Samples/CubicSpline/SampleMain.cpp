#include <Logging/SyrinxLogManager.h>
#include <Pipeline/SyrinxEngine.h>

#include "CubicSplineInterpolation.h"
#include "PolynomialInterpolation.h"
#include "PolynomialFitting.h"
#include "ParametricCurve.h"


void DrawPointsAndLines(const std::vector<Eigen::Vector2d>& points, bool drawLine = false)
{
    auto *x = new double[points.size()];
    auto *y = new double[points.size()];

    for (int i = 0; i < points.size(); ++ i) {
        x[i] = points[i].x();
        y[i] = points[i].y();
    }

    ImPlot::PlotScatter("points", x, y, points.size());
    if (drawLine) {
        ImPlot::PlotLine("line", x, y, points.size());
    }

    delete[] x;
    delete[] y;
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
    const char* methodName[] = {
            "Uniform-Polynomial",
            "Uniform-CubicSpline",
            "Uniform-BezierCurve",
            "Chordal-Polynomial",
            "Chordal-CubicSpline",
            "Chordal-BezierCurve"
    };

    auto uniformPolynomialParametricCurve = new ParametricCurve(ParametrizationMethod::Uniform, FittingMethod::PolynomialFitting, 0.0f);
    auto uniformCubicSplineParametricCurve = new ParametricCurve(ParametrizationMethod::Uniform, FittingMethod::CubicSpline, 0.0f);
    auto uniformBezierParametricCurve = new ParametricCurve(ParametrizationMethod::Chordal, FittingMethod::BezierCurve, 0.0f);
    auto chordalPolynomialParametricCurve = new ParametricCurve(ParametrizationMethod::Uniform, FittingMethod::PolynomialFitting, 0.0f);
    auto chordalCubicSplineParametricCurve = new ParametricCurve(ParametrizationMethod::Chordal, FittingMethod::CubicSpline, 0.0f);
    auto chordalBezierParametricCurve = new ParametricCurve(ParametrizationMethod::Chordal, FittingMethod::BezierCurve, 0.0f);
    std::vector<ParametricCurve*> parametricCurveList = {
            uniformPolynomialParametricCurve,
            uniformCubicSplineParametricCurve,
            uniformBezierParametricCurve,
            chordalPolynomialParametricCurve,
            chordalCubicSplineParametricCurve,
            chordalBezierParametricCurve
    };

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
        }

        if (ImGui::SliderFloat("Lambda", &lambda, 0.0, 1.0)) {
            for (auto parametricCurve : parametricCurveList) {
                parametricCurve->SetLambda(lambda);
                if (solved) {
                    parametricCurve->Solve(points);
                }
            }
        }
        ImGui::Spacing();

        static bool show = false;
        const auto axisFlags = ImPlotAxisFlags_Lock;
        if (ImPlot::BeginPlot("Plot", nullptr, nullptr, ImVec2(width / 1.5f, height / 1.5f), ImPlotFlags_None, axisFlags, axisFlags)) {
            if (ImPlot::IsPlotHovered() && ImGui::IsMouseDoubleClicked(0)) {
                ImPlotPoint p = ImPlot::GetPlotMousePos();
                points.emplace_back(p.x, p.y);
                SYRINX_INFO_FMT("x= {}, y={}", p.x, p.y);
            }
            DrawPointsAndLines(points, true);

            if (solved) {
                for (int methodIndex = 0; methodIndex < parametricCurveList.size(); ++ methodIndex) {
                    auto parametricCurve = parametricCurveList[methodIndex];

                    int count = 500;
                    auto x = new double[count];
                    auto y = new double[count];
                    for (int i = 0; i < count; ++i) {
                        double t = 0 + i * 1 / static_cast<double>(count - 1);
                        auto pos = parametricCurve->F(t);
                        x[i] = pos.x();
                        y[i] = pos.y();
                    }
                    ImPlot::PlotLine(methodName[methodIndex], x, y, count);
                }
            }
            ImPlot::EndPlot();
        }

        if (points.size() >= 2) {
            for (const auto parametricCurve : parametricCurveList) {
                parametricCurve->Solve(points);
            }
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