#include <Math/SyrinxMath.h>
#include <Logging/SyrinxLogManager.h>
#include <Pipeline/SyrinxDisplayDevice.h>
#include <ResourceManager/SyrinxModelManager.h>


int main(int argc, char *argv[])
{
    auto logManager = std::make_unique<Syrinx::LogManager>();
    Syrinx::DisplayDevice displayDevice;
    displayDevice.setMajorVersionNumber(4);
    displayDevice.setMinorVersionNumber(6);
    displayDevice.setDebugMessageHandler(Syrinx::DefaultDebugHandler);

    const int WINDOW_WIDTH = 800;
    const int WINDOW_HEIGHT = 800;
    auto renderWindow = displayDevice.createWindow("Load Model Sample", WINDOW_WIDTH, WINDOW_HEIGHT);

    Syrinx::FileManager fileManager;
    fileManager.addSearchPath("../SampleMedias/");

    Syrinx::HardwareResourceManager hardwareResourceManager;
    Syrinx::MeshManager meshManager(&fileManager, &hardwareResourceManager);
    Syrinx::ShaderManager shaderManager(&fileManager, &hardwareResourceManager);
    Syrinx::MaterialManager materialManager(&fileManager, &hardwareResourceManager, &shaderManager);
    Syrinx::ModelManager modelManager(&fileManager, &meshManager, &materialManager);

    shaderManager.addShaderSearchPath("../../Medias/Library");

    auto cubeModel = modelManager.createOrRetrieve("blue-cube.smodel");
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

            auto material = cubeModel->getMaterial("blue-unlit-color.smat");
            SYRINX_ASSERT(material);
            auto shaderVars = material->getShaderVars("constant-color.shader");
            SYRINX_ASSERT(shaderVars);

            auto shader = shaderVars->getShader();
            auto& vertexProgramVars = *shaderVars->getProgramVars(Syrinx::ProgramStageType::VertexStage);
            auto& fragmentProgramVars = *shaderVars->getProgramVars(Syrinx::ProgramStageType::FragmentStage);

            auto programPipeline = shader.getProgramPipeline();
            auto vertexProgram = shader.getShaderModule(Syrinx::ProgramStageType::VertexStage);
            auto fragmentProgram = shader.getShaderModule(Syrinx::ProgramStageType::FragmentStage);

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

            vertexProgramVars["SyrinxMatrixBuffer"]["SYRINX_MATRIX_PROJ"] = projectionMatrix;
            vertexProgramVars["SyrinxMatrixBuffer"]["SYRINX_MATRIX_VIEW"] = viewMatrix;
            vertexProgramVars["SyrinxMatrixBuffer"]["SYRINX_MATRIX_WORLD"] = modelMatrix;

            vertexProgram->updateProgramVars(vertexProgramVars);
            vertexProgram->uploadParametersToGpu();
            vertexProgram->bindResources();
            fragmentProgram->updateProgramVars(fragmentProgramVars);
            fragmentProgram->uploadParametersToGpu();
            fragmentProgram->bindResources();

            glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(3 * mesh->getNumTriangle()), GL_UNSIGNED_INT, nullptr);
            glBindVertexArray(0);
        }
        renderWindow->swapBuffer();
    }
    return 0;
}