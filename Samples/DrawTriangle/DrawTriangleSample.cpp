#include <GL/glew.h>
#include <Logging/SyrinxLogManager.h>
#include <Pipeline/SyrinxDisplayDevice.h>
#include <Pipeline/SyrinxEngineSetting.h>
#include <HardwareResource/SyrinxProgramStage.h>
#include <HardwareResource/SyrinxProgramPipeline.h>
#include <HardwareResource/SyrinxHardwareVertexBuffer.h>
#include <HardwareResource/SyrinxHardwareIndexBuffer.h>
#include <HardwareResource/SyrinxVertexInputState.h>


int main(int argc, char *argv[])
{
    Syrinx::LogManager *logManager = new Syrinx::LogManager();
    Syrinx::DisplayDevice displayDevice;

    displayDevice.setMajorVersionNumber(4);
    displayDevice.setMinorVersionNumber(5);
    displayDevice.setDebugMessageHandler(Syrinx::DefaultDebugHandler);
    auto renderWindow = displayDevice.createWindow("Draw Triangle Sample", 800, 600);

    const std::string vertexProgramSource =
        "#version 450 core\n"
        "layout(location = 0) in vec3 aPos;\n"
        "out gl_PerVertex {\n"
        "    vec4 gl_Position;\n"
        "};\n"
        "void main() {\n"
        "    gl_Position = vec4(aPos, 1.0);\n"
        "}\n";

    const std::string fragmentProgramSource =
        "#version 450 core\n"
        "out vec4 FragColor;\n"
        "void main()\n"
        "{\n"
        "   FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);\n"
        "}\n";

    auto vertexProgram = std::make_unique<Syrinx::ProgramStage>("draw triangle vertex program");
    vertexProgram->setType(Syrinx::ProgramStageType::VertexStage);
    vertexProgram->setSource(vertexProgramSource);
    vertexProgram->create();

    auto fragmentProgram = std::make_unique<Syrinx::ProgramStage>("draw triangle fragment program");
    fragmentProgram->setType(Syrinx::ProgramStageType::FragmentStage);
    fragmentProgram->setSource(fragmentProgramSource);
    fragmentProgram->create();

    Syrinx::ProgramPipeline programPipeline("draw triangle program pipeline");
    programPipeline.create();
    programPipeline.bindProgramStage(vertexProgram.get());
    programPipeline.bindProgramStage(fragmentProgram.get());

    float vertices[] = {
        -0.5f, -0.5f, 0.0f, // left
        0.5f, -0.5f, 0.0f,  // right
        0.0f,  0.5f, 0.0f   // top
    };

    uint16_t indices[] = {
        0, 1, 2
    };

    auto vertexBuffer = std::make_unique<Syrinx::HardwareBuffer>("triangle vertex buffer");
    Syrinx::HardwareVertexBuffer hardwareVertexBuffer(std::move(vertexBuffer));
    hardwareVertexBuffer.setVertexNumber(3);
    hardwareVertexBuffer.setVertexSizeInBytes(3 * sizeof(float));
    hardwareVertexBuffer.setData(vertices);
    hardwareVertexBuffer.create();

    auto indexBuffer = std::make_unique<Syrinx::HardwareBuffer>("triangle index buffer");
    Syrinx::HardwareIndexBuffer hardwareIndexBuffer(std::move(indexBuffer));
    hardwareIndexBuffer.setIndexType(Syrinx::IndexType::UINT16);
    hardwareIndexBuffer.setIndexNumber(3);
    hardwareIndexBuffer.setData(indices);
    hardwareIndexBuffer.create();

    Syrinx::VertexAttributeDescription vertexAttributeDescription(0, Syrinx::VertexAttributeSemantic::Position, Syrinx::VertexAttributeDataType::FLOAT3);
    Syrinx::VertexDataDescription vertexDataDescription(&hardwareVertexBuffer, 0, 0, 3 * sizeof(float));

    Syrinx::VertexInputState vertexInputState("triangle vertex input state");
    vertexInputState.addVertexAttributeDescription(vertexAttributeDescription);
    vertexInputState.addVertexDataDescription(vertexDataDescription);
    vertexInputState.addIndexBuffer(&hardwareIndexBuffer);
    vertexInputState.create();

    while (renderWindow->isOpen()) {
        float defaultValueForColorAttachment[] = {0.5, 0.0, 0.5, 1.0};
        glClearNamedFramebufferfv(0, GL_COLOR, 0, defaultValueForColorAttachment);
        float defaultDepthValue = 1.0;
        glClearNamedFramebufferfv(0, GL_DEPTH, 0, &defaultDepthValue);

        glBindVertexArray(vertexInputState.getHandle());
        glBindProgramPipeline(programPipeline.getHandle());
        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, nullptr);

        renderWindow->swapBuffer();
    }

    return 0;
}