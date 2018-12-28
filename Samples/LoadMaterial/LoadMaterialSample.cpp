#include <Math/SyrinxMath.h>
#include <Logging/SyrinxLogManager.h>
#include <RenderPipeline/SyrinxDisplayDevice.h>
#include <ResourceManager/SyrinxMeshManager.h>
#include <ResourceManager/SyrinxMaterialManager.h>


int main(int argc, char *argv[])
{
    Syrinx::LogManager *logManager = new Syrinx::LogManager();
    Syrinx::DisplayDevice displayDevice;
    displayDevice.setMajorVersionNumber(4);
    displayDevice.setMinorVersionNumber(5);
    displayDevice.setDebugMessageHandler(Syrinx::DefaultDebugHandler);

    const int WINDOW_WIDTH = 800;
    const int WINDOW_HEIGHT = 800;
    auto renderWindow = displayDevice.createWindow("Load Material Sample", WINDOW_WIDTH, WINDOW_HEIGHT);

    Syrinx::FileManager fileManager;
    Syrinx::HardwareResourceManager hardwareResourceManager(&fileManager);
    Syrinx::MeshManager meshManager(&fileManager, &hardwareResourceManager);
    Syrinx::MaterialManager materialManager(&fileManager, &hardwareResourceManager);

    fileManager.addSearchPath("../../Medias/");
    auto cubeMesh = meshManager.createMesh("cube.smesh");
    auto cubeMaterial = materialManager.createMaterial("display-red-color.smat");

    auto displayColorShader = cubeMaterial->getShader();
    auto lightingPass = displayColorShader->getShaderPass("lighting");
    SYRINX_ASSERT(lightingPass);
    auto vertexProgramForLightingPass = lightingPass->getProgramStage(Syrinx::ProgramStageType::VertexStage);
    auto fragmentProgramForLightingPass = lightingPass->getProgramStage(Syrinx::ProgramStageType::FragmentStage);
    auto lightingPassProgramPipeline = lightingPass->getProgramPipeline();

    Syrinx::ShaderParameter* colorDisplayedParameter = displayColorShader->getShaderParameter("displayColor");
    SYRINX_ASSERT(colorDisplayedParameter);
    Syrinx::Color displayColor = std::get<Syrinx::Color>(colorDisplayedParameter->getValue());
    fragmentProgramForLightingPass->updateParameter(colorDisplayedParameter->getName(), displayColor);

    while (renderWindow->isOpen()) {
        float defaultValueForColorAttachment[] = {0.0, 0.0, 1.0, 1.0};
        glClearNamedFramebufferfv(0, GL_COLOR, 0, defaultValueForColorAttachment);
        float defaultDepthValue = 1.0;
        glClearNamedFramebufferfv(0, GL_DEPTH, 0, &defaultDepthValue);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);

        glBindProgramPipeline(lightingPassProgramPipeline->getHandle());

        Syrinx::Matrix4x4 modelMatrix(1.0f);
        modelMatrix = glm::rotate(modelMatrix, glm::radians(45.0f), Syrinx::Vector3f(0.0f, 1.0f, 0.0f));
        Syrinx::Matrix4x4 projectionMatrix = glm::perspective(glm::radians(45.0f), (GLfloat)WINDOW_WIDTH / (GLfloat)WINDOW_HEIGHT, 0.1f, 100.0f);
        Syrinx::Vector3f cameraPos = Syrinx::Vector3f(0.0f, 0.0f, 3.0f);
        Syrinx::Vector3f cameraFront = Syrinx::Vector3f(0.0f, 0.0f, -1.0f);
        Syrinx::Vector3f upVector = Syrinx::Vector3f(0.0f, 1.0f, 0.0f);
        Syrinx::Matrix4x4 viewMatrix = glm::lookAt(cameraPos, cameraPos + cameraFront, upVector);
        vertexProgramForLightingPass->updateParameter("uModelMatrix", modelMatrix);
        vertexProgramForLightingPass->updateParameter("uViewMatrix", viewMatrix);
        vertexProgramForLightingPass->updateParameter("uProjectionMatrix", projectionMatrix);

        glBindVertexArray(cubeMesh->getVertexInputState().getHandle());
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(3 * cubeMesh->getNumTriangle()), GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);

        renderWindow->swapBuffer();
    }
    return 0;
}