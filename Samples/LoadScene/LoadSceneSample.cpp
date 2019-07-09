#include <Math/SyrinxMath.h>
#include <Component/SyrinxRenderer.h>
#include <Logging/SyrinxLogManager.h>
#include <Pipeline/SyrinxDisplayDevice.h>
#include <Scene/SyrinxSceneManager.h>


int main(int argc, char *argv[])
{
    auto logManager = std::make_unique<Syrinx::LogManager>();
    Syrinx::DisplayDevice displayDevice;
    displayDevice.setMajorVersionNumber(4);
    displayDevice.setMinorVersionNumber(6);
    displayDevice.setDebugMessageHandler(Syrinx::DefaultDebugHandler);

    const int WIDTH = 800;
    const int HEIGHT = 800;
    auto renderWindow = displayDevice.createWindow("Load Scene Sample", WIDTH, HEIGHT);

    Syrinx::FileManager fileManager;
    Syrinx::HardwareResourceManager hardwareResourceManager;
    Syrinx::ShaderManager shaderManager(&fileManager, &hardwareResourceManager);
    Syrinx::MaterialManager materialManager(&fileManager, &hardwareResourceManager, &shaderManager);
    Syrinx::MeshManager meshManager(&fileManager, &hardwareResourceManager);
    Syrinx::ModelManager modelManager(&fileManager, &meshManager, &materialManager);
    Syrinx::SceneManager sceneManager(&fileManager, &modelManager);

    fileManager.addSearchPath("../SampleMedias/");
    shaderManager.addShaderSearchPath("../../Medias/Library");

    const std::string sceneFile = "cube-scene.scene";
    auto scene = sceneManager.importScene(sceneFile);
    auto cubeModelNode = scene->findSceneNode("cube-entity");
    auto cubeModelEntity = cubeModelNode->getEntity();
    SYRINX_ASSERT(cubeModelEntity);

    auto cubeMeshNode = cubeModelNode->getChild("cube.smesh");
    SYRINX_ASSERT(cubeMeshNode);
    auto cubeEntity = cubeMeshNode->getEntity();

    SYRINX_ASSERT(cubeEntity);
    auto& cubeRenderer = cubeEntity->getComponent<Syrinx::Renderer>();
    auto cubeMesh = cubeRenderer.getMesh();
    auto cubeMaterial = cubeRenderer.getMaterial();

    auto constantColorShaderVar = cubeMaterial->getShaderVars("constant-color.shader");
    SYRINX_ASSERT(constantColorShaderVar);

    auto shader = constantColorShaderVar->getShader();
    auto programPipeline = shader.getProgramPipeline();
    auto vertexProgram = shader.getShaderModule(Syrinx::ProgramStageType::VertexStage);
    auto fragmentProgram = shader.getShaderModule(Syrinx::ProgramStageType::FragmentStage);
    auto& vertexProgramVars = *constantColorShaderVar->getProgramVars(Syrinx::ProgramStageType::VertexStage);
    auto& fragmentProgramVars = *constantColorShaderVar->getProgramVars(Syrinx::ProgramStageType::FragmentStage);

    while (renderWindow->isOpen()) {
        float defaultValueForColorAttachment[] = {0.5, 0.0, 0.5, 1.0};
        glClearNamedFramebufferfv(0, GL_COLOR, 0, defaultValueForColorAttachment);
        float defaultDepthValue = 1.0;
        glClearNamedFramebufferfv(0, GL_DEPTH, 0, &defaultDepthValue);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);

        glBindProgramPipeline(programPipeline->getHandle());
        Syrinx::Matrix4x4 modelMatrix(1.0f);
        modelMatrix = glm::rotate(modelMatrix, glm::radians(45.0f), Syrinx::Vector3f(0.0, 1.0, 0.0));
        Syrinx::Matrix4x4 projectionMatrix = glm::perspective(glm::radians(45.0f),
                                                              static_cast<float>(WIDTH) / static_cast<float>(HEIGHT),
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

        GLuint vertexInputStateHandle = cubeMesh->getVertexInputState().getHandle();
        glBindVertexArray(vertexInputStateHandle);
        glDrawElements(GL_TRIANGLES, static_cast<int>(cubeMesh->getNumTriangle() * 3), GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);

        renderWindow->swapBuffer();
    }
    return 0;
}
