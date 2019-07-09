#include <Math/SyrinxMath.h>
#include <Logging/SyrinxLogManager.h>
#include <Pipeline/SyrinxDisplayDevice.h>
#include <ResourceManager/SyrinxMeshManager.h>
#include <ResourceManager/SyrinxMaterialManager.h>


int main(int argc, char *argv[])
{
    auto logManager = std::make_unique<Syrinx::LogManager>();

    Syrinx::DisplayDevice displayDevice;
    displayDevice.setMajorVersionNumber(4);
    displayDevice.setMinorVersionNumber(6);
    displayDevice.setDebugMessageHandler(Syrinx::DefaultDebugHandler);

    const int WINDOW_WIDTH = 800;
    const int WINDOW_HEIGHT = 800;
    auto renderWindow = displayDevice.createWindow("Load Material Sample", WINDOW_WIDTH, WINDOW_HEIGHT);

    Syrinx::FileManager fileManager;
    Syrinx::HardwareResourceManager hardwareResourceManager;
    Syrinx::MeshManager meshManager(&fileManager, &hardwareResourceManager);
    Syrinx::ShaderManager shaderManager(&fileManager, &hardwareResourceManager);
    Syrinx::MaterialManager materialManager(&fileManager, &hardwareResourceManager, &shaderManager);

    shaderManager.addShaderSearchPath("../../Medias/Library");

    fileManager.addSearchPath(".");
    fileManager.addSearchPath("../SampleMedias/");
    auto cubeMesh = meshManager.createOrRetrieve("cube.smesh");
    auto cubeMaterial = materialManager.createOrRetrieve("blue-unlit-color.smat");
    SYRINX_ASSERT(cubeMesh);
    SYRINX_ASSERT(cubeMaterial);

    auto shaderVars = cubeMaterial->getShaderVars("constant-color.shader");
    SYRINX_ASSERT(shaderVars);
    auto& vertexVars = *(shaderVars->getProgramVars(Syrinx::ProgramStageType::VertexStage));
    auto& programVars = *(shaderVars->getProgramVars(Syrinx::ProgramStageType::FragmentStage));

    auto& shader = shaderVars->getShader();
    auto programPipeline = shader.getProgramPipeline();
    auto vertexProgram = shader.getShaderModule(Syrinx::ProgramStageType::VertexStage);
    auto fragmentProgram = shader.getShaderModule(Syrinx::ProgramStageType::FragmentStage);

    fragmentProgram->updateProgramVars(programVars);
    fragmentProgram->uploadParametersToGpu();

    while (renderWindow->isOpen()) {
        float defaultValueForColorAttachment[] = {0.0, 0.0, 1.0, 1.0};
        glClearNamedFramebufferfv(0, GL_COLOR, 0, defaultValueForColorAttachment);
        float defaultDepthValue = 1.0;
        glClearNamedFramebufferfv(0, GL_DEPTH, 0, &defaultDepthValue);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);

        glBindProgramPipeline(programPipeline->getHandle());

        Syrinx::Matrix4x4 modelMatrix(1.0f);
        modelMatrix = glm::rotate(modelMatrix, glm::radians(45.0f), Syrinx::Vector3f(0.0f, 1.0f, 0.0f));
        Syrinx::Matrix4x4 projectionMatrix = glm::perspective(glm::radians(45.0f), (GLfloat)WINDOW_WIDTH / (GLfloat)WINDOW_HEIGHT, 0.1f, 100.0f);
        Syrinx::Vector3f cameraPos = Syrinx::Vector3f(0.0f, 0.0f, 3.0f);
        Syrinx::Vector3f cameraFront = Syrinx::Vector3f(0.0f, 0.0f, -1.0f);
        Syrinx::Vector3f upVector = Syrinx::Vector3f(0.0f, 1.0f, 0.0f);
        Syrinx::Matrix4x4 viewMatrix = glm::lookAt(cameraPos, cameraPos + cameraFront, upVector);

        vertexVars["SyrinxMatrixBuffer"]["SYRINX_MATRIX_PROJ"] = projectionMatrix;
        vertexVars["SyrinxMatrixBuffer"]["SYRINX_MATRIX_VIEW"] = viewMatrix;
        vertexVars["SyrinxMatrixBuffer"]["SYRINX_MATRIX_WORLD"] = modelMatrix;
        vertexProgram->updateProgramVars(vertexVars);
        vertexProgram->uploadParametersToGpu();
        vertexProgram->bindResources();

        fragmentProgram->bindResources();

        glBindVertexArray(cubeMesh->getVertexInputState().getHandle());
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(3 * cubeMesh->getNumTriangle()), GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);

        renderWindow->swapBuffer();
    }
    return 0;
}