#include <Math/SyrinxMath.h>
#include <Logging/SyrinxLogManager.h>
#include <ResourceManager/SyrinxMeshManager.h>
#include <RenderPipeline/SyrinxEngineSetting.h>
#include <RenderPipeline/SyrinxDisplayDevice.h>
#include <HardwareResource/SyrinxProgramStage.h>
#include <HardwareResource/SyrinxProgramPipeline.h>


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

    const std::string vertexShaderSource =
            "#version 450 core\n"
            "uniform mat4 uModelMatrix;\n"
            "uniform mat4 uViewMatrix;\n"
            "uniform mat4 uProjectionMatrix;\n"
            "layout (location = 0) in vec3 _inVertexPosition;\n"
            "layout (location = 1) in vec3 _inVertexNormal;\n"
            "layout (location = 2) in vec2 _inVertexTexCoord;\n"
            "out vec3 _NormalW;\n"
            "out vec2 _TexCoord;\n"
            "out gl_PerVertex\n"
            "{\n"
            "    vec4 gl_Position;\n"
            "    float gl_PointSize;\n"
            "    float gl_ClipDistance[];\n"
            "};\n"
            "void main()\n"
            "{\n"
            "gl_Position = uProjectionMatrix * uViewMatrix * uModelMatrix * vec4(_inVertexPosition, 1.0);\n"
            "_NormalW = normalize(transpose(inverse(mat3(uModelMatrix))) * _inVertexNormal);\n"
            "_TexCoord = _inVertexTexCoord;\n"
            "}\n";


    const std::string fragmentShaderSource =
            "#version 450 core\n"
            "in vec3 _NormalW;\n"
            "out vec4 outFragColor;\n"
            "void main()\n"
            "{"
            "    float NdotL = dot(_NormalW, vec3(0.0, 1.0, 0.0));\n"
            "    float lightInfluence = NdotL * 0.5 + 0.5;\n"
            "    vec3 result = mix(vec3(0.0), vec3(0.5, 0.0, 0.5), lightInfluence);\n"
            "    outFragColor = vec4(result, 1.0);\n"
            "}";


    Syrinx::ProgramStage vertexProgram("model vertex program");
    vertexProgram.setType(Syrinx::ProgramStageType::VertexStage);
    vertexProgram.setSource(vertexShaderSource);
    vertexProgram.create();

    Syrinx::ProgramStage fragmentProgram("model fragment program");
    fragmentProgram.setType(Syrinx::ProgramStageType::FragmentStage);
    fragmentProgram.setSource(fragmentShaderSource);
    fragmentProgram.create();

    Syrinx::ProgramPipeline programPipeline("draw model program pipeline");
    programPipeline.create();
    programPipeline.bindProgramStage(&vertexProgram);
    programPipeline.bindProgramStage(&fragmentProgram);

    Syrinx::FileManager fileManager;
    fileManager.addSearchPath("../../Medias/Models/");
    Syrinx::HardwareResourceManager hardwareResourceManager(&fileManager);
    Syrinx::MeshManager meshManager(&fileManager, &hardwareResourceManager);
    auto cubeMesh = meshManager.createMesh("cube.smesh");
    while (renderWindow->isOpen()) {
        float defaultValueForColorAttachment[] = {0.0, 0.0, 1.0, 1.0};
        glClearNamedFramebufferfv(0, GL_COLOR, 0, defaultValueForColorAttachment);
        float defaultDepthValue = 1.0;
        glClearNamedFramebufferfv(0, GL_DEPTH, 0, &defaultDepthValue);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);

        glBindProgramPipeline(programPipeline.getHandle());
        Syrinx::Matrix4x4 modelMatrix(1.0f);
        modelMatrix = glm::rotate(modelMatrix, glm::radians(45.0f), Syrinx::Vector3f(0.0, 1.0, 0.0));
        Syrinx::Matrix4x4 projectionMatrix = glm::perspective(glm::radians(45.0f),
                                                              static_cast<float>(WINDOW_WIDTH) / static_cast<float>(WINDOW_HEIGHT),
                                                              0.1f,
                                                              100.0f);
        Syrinx::Vector3f cameraPos = {0.0f, 0.0f, 3.0f};
        Syrinx::Vector3f cameraFront = {0.0f, 0.0f, -1.0f};
        Syrinx::Vector3f upVector = {0.0f, 1.0f, 0.0f};
        Syrinx::Matrix4x4 viewMatrix = glm::lookAt(cameraPos, cameraPos + cameraFront, upVector);

        vertexProgram.updateParameter("uModelMatrix", modelMatrix);
        vertexProgram.updateParameter("uViewMatrix", viewMatrix);
        vertexProgram.updateParameter("uProjectionMatrix", projectionMatrix);

        glBindVertexArray(cubeMesh->getVertexInputState().getHandle());
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(3 * cubeMesh->getNumTriangle()), GL_UNSIGNED_INT, nullptr);
        renderWindow->swapBuffer();
    }
    return 0;
}