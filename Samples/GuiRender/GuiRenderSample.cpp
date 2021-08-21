#include <Logging/SyrinxLogManager.h>
#include <Pipeline/SyrinxEngine.h>

#include "PolynomialInterpolation.h"
#include "GaussianInterpolation.h"
#include "PolynomialFitting.h"


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

    float sigma = 1.0f;
    float lambda = 1.0f;

    auto *polynomialInterpolation = new PolynomialInterpolation();
    auto *polynomialFitting = new PolynomialFitting(0.0f);
    auto *ridgeRegressionFitting = new PolynomialFitting(1.0f);
    auto *gaussianInterpolation = new GaussianInterpolation();

    std::vector<INumericalMethod*> methods{polynomialInterpolation, gaussianInterpolation, polynomialFitting, ridgeRegressionFitting};
    std::vector<std::string> methodNames = {
            "Polynomial Interpolation",
            "Gaussian Interpolation",
            "Polynomial Fitting",
            "Ridge Regression"};

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
        ImGui::SliderFloat("Sigma", &sigma, 0.0, 1.0f);
        ImGui::Spacing();

        if (ImGui::SliderFloat("Lambda", &lambda, 0.0, 1.0)) {
            if (solved) {
                ridgeRegressionFitting = new PolynomialFitting(lambda);
                ridgeRegressionFitting->Solve(points);
                methods[3] = ridgeRegressionFitting;
            }
        }
        ImGui::Spacing();

        gaussianInterpolation->SetSigma(sigma);
        ridgeRegressionFitting->SetLambda(lambda);

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
                int count = 100;

                auto x = new double[count];
                std::vector<double*> results;
                for (int methodIndex = 0; methodIndex < methods.size(); ++ methodIndex) {
                    auto y = new double[count];
                    results.push_back(y);
                }

                for (int i = 0; i < count; ++ i) {
                    x[i] = 0 + i * 1 / static_cast<double>(count);

                    for (int methodIndex = 0; methodIndex < methods.size(); ++ methodIndex) {
                        results[methodIndex][i] = methods[methodIndex]->F(x[i]);
                    }
                }

                for (int i = 0; i < methodNames.size(); ++ i) {
                    const auto& methodName = methodNames[i];
                    ImPlot::PlotLine(methodName.c_str(), x, results[i], count);
                }
            }
            ImPlot::EndPlot();
        }

        if (points.size() > 1 || ImGui::Button("Calculation")) {
            for (auto method : methods) {
                method->Solve(points);
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