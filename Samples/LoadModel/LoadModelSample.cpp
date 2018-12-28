#include <Math/SyrinxMath.h>
#include <RenderPipeline/SyrinxDisplayDevice.h>
#include <Logging/SyrinxLogManager.h>
#include <ResourceManager/SyrinxModelManager.h>


int main(int argc, char *argv[])
{
    Syrinx::LogManager *logManager = new Syrinx::LogManager();
    Syrinx::DisplayDevice displayDevice;
    displayDevice.setMajorVersionNumber(4);
    displayDevice.setMinorVersionNumber(5);
    displayDevice.setDebugMessageHandler(Syrinx::DefaultDebugHandler);

    const int WINDOW_WIDTH = 800;
    const int WINDOW_HEIGHT = 800;
    auto renderWindow = displayDevice.createWindow("Load Model Sample", WINDOW_WIDTH, WINDOW_HEIGHT);

    Syrinx::FileManager fileManager;
    fileManager.addSearchPath("../../Medias");
    Syrinx::HardwareResourceManager hardwareResourceManager(&fileManager);
    Syrinx::MeshManager meshManager(&fileManager, &hardwareResourceManager);
    Syrinx::MaterialManager materialManager(&fileManager, &hardwareResourceManager);
    Syrinx::ModelManager modelManager(&fileManager, &meshManager, &materialManager);
    auto cubeModel = modelManager.createModel("cube.smodel");

    while (renderWindow->isOpen()) {
        float defaultValueForColorAttachment[] = {0.0, 0.0, 1.0, 1.0};
        glClearNamedFramebufferfv(0, GL_COLOR, 0, defaultValueForColorAttachment);
        float defaultDepthValue = 1.0;
        glClearNamedFramebufferfv(0, GL_DEPTH, 0, &defaultDepthValue);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);

        const auto& meshList = cubeModel->getMeshList();
        for (const auto mesh : meshList) {
            GLuint vertexInputStateHandle = mesh->getVertexInputState().getHandle();
            glBindVertexArray(vertexInputStateHandle);

            const Syrinx::Material* material = cubeModel->getMaterial("display-red-color");
            auto lightingShaderPass = material->getShader()->getShaderPass("lighting");
            SYRINX_ASSERT(lightingShaderPass);
            auto programPipeline = lightingShaderPass->getProgramPipeline();
            auto vertexProgram = lightingShaderPass->getProgramStage(Syrinx::ProgramStageType::VertexStage);
            auto fragmentProgram = lightingShaderPass->getProgramStage(Syrinx::ProgramStageType::FragmentStage);

            glBindProgramPipeline(programPipeline->getHandle());
            Syrinx::Matrix4x4 modelMatrix = Syrinx::Matrix4x4(1.0f);
            modelMatrix = glm::rotate(modelMatrix, glm::radians(45.0f), Syrinx::Vector3f(0.0, 1.0, 0.0));
            Syrinx::Matrix4x4 projectionMatrix = glm::perspective(glm::radians(45.0f),
                                                                  static_cast<float>(WINDOW_WIDTH) / static_cast<float>(WINDOW_HEIGHT),
                                                                  0.1f,
                                                                  100.0f);
            Syrinx::Vector3f cameraPos = {0.0f, 0.0f, 3.0f};
            Syrinx::Vector3f cameraFront = {0.0f, 0.0f, -1.0f};
            Syrinx::Vector3f upVector = {0.0f, 1.0f, 0.0f};
            Syrinx::Matrix4x4 viewMatrix = glm::lookAt(cameraPos, cameraPos + cameraFront, upVector);

            vertexProgram->updateParameter("uModelMatrix", modelMatrix);
            vertexProgram->updateParameter("uViewMatrix", viewMatrix);
            vertexProgram->updateParameter("uProjectionMatrix", projectionMatrix);

            auto displayColorParameter = material->getShader()->getShaderParameter("displayColor");
            fragmentProgram->updateParameter(displayColorParameter->getName(), std::get<Syrinx::Color>(displayColorParameter->getValue()));
            glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(3 * mesh->getNumTriangle()), GL_UNSIGNED_INT, nullptr);
            glBindVertexArray(0);
        }
        renderWindow->swapBuffer();
    }
    return 0;
}