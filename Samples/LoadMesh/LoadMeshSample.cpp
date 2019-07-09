#include <Math/SyrinxMath.h>
#include <Logging/SyrinxLogManager.h>
#include <Program/SyrinxProgramCompiler.h>
#include <HardwareResource/SyrinxProgramStage.h>
#include <HardwareResource/SyrinxProgramPipeline.h>
#include <Pipeline/SyrinxEngineSetting.h>
#include <Pipeline/SyrinxDisplayDevice.h>
#include <ResourceManager/SyrinxMeshManager.h>


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

    const std::string vertexShaderSource =
        "layout (std140, binding = 0) uniform Matrix { \n"
        "    mat4 uModelMatrix;\n"
        "    mat4 uViewMatrix;\n"
        "    mat4 uProjectionMatrix;\n"
        "};\n"
        "layout (location = 0) in vec3 _inVertexPosition;\n"
        "layout (location = 1) in vec3 _inVertexNormal;\n"
        "layout (location = 2) in vec2 _inVertexTexCoord;\n"
        "layout (location = 0) out gl_PerVertex {\n"
        "    vec4 gl_Position;\n"
        "};\n"
        "layout (location = 1) out vec3 _NormalW;\n"
        "layout (location = 2) out vec2 _TexCoord;\n"
        "void main(){\n"
        "gl_Position = uProjectionMatrix * uViewMatrix * uModelMatrix * vec4(_inVertexPosition, 1.0);\n"
        "_NormalW = normalize(transpose(inverse(mat3(uModelMatrix))) * _inVertexNormal);\n"
        "_TexCoord = _inVertexTexCoord;\n"
        "}\n";


    const std::string fragmentShaderSource =
        "layout(location = 1) in vec3 _NormalW;\n"
        "layout(location = 0) out vec4 outFragColor;\n"
        "void main()\n"
        "{\n"
        "    vec3 lightDir = normalize(vec3(1.0, 1.0, 0.0));\n"
        "    float NdotL = dot(_NormalW, lightDir);\n"
        "    float lightInfluence = NdotL * 0.5 + 0.5;\n"
        "    vec3 result = mix(vec3(0.0), vec3(0.5, 0.0, 0.5), lightInfluence);\n"
        "    outFragColor = vec4(result, 1.0);\n"
        "}";

    Syrinx::ProgramCompiler compiler;
    Syrinx::HardwareResourceManager hardwareResourceManager;
    auto vertexProgramBinarySource = compiler.compile("vertex", vertexShaderSource, Syrinx::ProgramStageType::VertexStage);
    auto vertexProgram = hardwareResourceManager.createProgramStage("model vertex program",
                                                                    std::move(vertexProgramBinarySource),
                                                                    Syrinx::ProgramStageType::VertexStage);

    auto fragmentProgramBinarySource = compiler.compile("fragment", fragmentShaderSource, Syrinx::ProgramStageType::FragmentStage);
    auto fragmentProgram = hardwareResourceManager.createProgramStage("model fragment program",
                                                                      std::move(fragmentProgramBinarySource),
                                                                      Syrinx::ProgramStageType::FragmentStage);

    auto programPipeline = hardwareResourceManager.createProgramPipeline("draw model program pipeline");
    programPipeline->bindProgramStage(vertexProgram);
    programPipeline->bindProgramStage(fragmentProgram);

    auto& vertexProgramVars = *vertexProgram->getProgramVars();
    Syrinx::FileManager fileManager;
    fileManager.addSearchPath("../SampleMedias/");
    Syrinx::MeshManager meshManager(&fileManager, &hardwareResourceManager);
    auto cubeMesh = meshManager.createOrRetrieve("cube.smesh");
    while (renderWindow->isOpen()) {
        float defaultValueForColorAttachment[] = {0.0, 0.0, 1.0, 1.0};
        glClearNamedFramebufferfv(0, GL_COLOR, 0, defaultValueForColorAttachment);
        float defaultDepthValue = 1.0;
        glClearNamedFramebufferfv(0, GL_DEPTH, 0, &defaultDepthValue);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);

        glBindProgramPipeline(programPipeline->getHandle());
        Syrinx::Matrix4x4 modelMatrix(1.0);
        modelMatrix = glm::rotate(modelMatrix, glm::radians(45.0f), Syrinx::Vector3f(0.0, 1.0, 0.0));
        Syrinx::Matrix4x4 projectionMatrix = glm::perspective(glm::radians(45.0f),
                                                              static_cast<float>(WINDOW_WIDTH) / static_cast<float>(WINDOW_HEIGHT),
                                                              0.1f,
                                                              100.0f);
        Syrinx::Vector3f cameraPos = {0.0f, 0.0f, 3.0f};
        Syrinx::Vector3f cameraFront = {0.0f, 0.0f, -1.0f};
        Syrinx::Vector3f upVector = {0.0f, 1.0f, 0.0f};
        Syrinx::Matrix4x4 viewMatrix = glm::lookAt(cameraPos, cameraPos + cameraFront, upVector);

        vertexProgramVars["Matrix"]["uModelMatrix"] = modelMatrix;
        vertexProgramVars["Matrix"]["uViewMatrix"] = viewMatrix;
        vertexProgramVars["Matrix"]["uProjectionMatrix"] = projectionMatrix;
        vertexProgram->updateProgramVars(vertexProgramVars);
        vertexProgram->uploadParametersToGpu();
        vertexProgram->bindResources();

        glBindVertexArray(cubeMesh->getVertexInputState().getHandle());
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(3 * cubeMesh->getNumTriangle()), GL_UNSIGNED_INT, nullptr);
        renderWindow->swapBuffer();
    }
    return 0;
}