#include <Logging/SyrinxLogManager.h>
#include <Pipeline/SyrinxDisplayDevice.h>
#include <Pipeline/SyrinxEngineSetting.h>
#include <Program/SyrinxProgramCompiler.h>
#include <Manager/SyrinxHardwareResourceManager.h>
#include <HardwareResource/SyrinxProgramStage.h>
#include <HardwareResource/SyrinxProgramPipeline.h>
#include <HardwareResource/SyrinxHardwareVertexBuffer.h>
#include <HardwareResource/SyrinxHardwareIndexBuffer.h>
#include <HardwareResource/SyrinxVertexInputState.h>


int main(int argc, char *argv[])
{
    auto logManager = std::make_unique<Syrinx::LogManager>();
    Syrinx::DisplayDevice displayDevice;

    displayDevice.setMajorVersionNumber(4);
    displayDevice.setMinorVersionNumber(6);
    displayDevice.setDebugMessageHandler(Syrinx::DefaultDebugHandler);
    auto renderWindow = displayDevice.createWindow("Draw Triangle Sample", 800, 600);

    const std::string vertexProgramSource =
        "layout(location = 0) in vec3 aPos;\n"
        "out gl_PerVertex {\n"
        "    vec4 gl_Position;\n"
        "};\n"
        "void main() {\n"
        "    gl_Position = vec4(aPos, 1.0);\n"
        "}\n";

    const std::string fragmentProgramSource =
        "layout(location = 0) out vec4 FragColor;\n"
        "void main()\n"
        "{\n"
        "   FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);\n"
        "}\n";

    Syrinx::ProgramCompiler compiler;
    Syrinx::HardwareResourceManager hardwareResourceManager;
    auto vertexProgramBinarySource = compiler.compile("vertex", vertexProgramSource, Syrinx::ProgramStageType::VertexStage);
    auto vertexProgram = hardwareResourceManager.createProgramStage("model vertex program",
                                                                    std::move(vertexProgramBinarySource),
                                                                    Syrinx::ProgramStageType::VertexStage);

    auto fragmentProgramBinarySource = compiler.compile("fragment", fragmentProgramSource, Syrinx::ProgramStageType::FragmentStage);
    auto fragmentProgram = hardwareResourceManager.createProgramStage("model fragment program",
                                                                      std::move(fragmentProgramBinarySource),
                                                                      Syrinx::ProgramStageType::FragmentStage);

    auto programPipeline = hardwareResourceManager.createProgramPipeline("draw model program pipeline");
    programPipeline->bindProgramStage(vertexProgram);
    programPipeline->bindProgramStage(fragmentProgram);


    float vertices[] = {
        -0.5f, -0.5f, 0.0f,
        0.5f, -0.5f, 0.0f,
        0.0f,  0.5f, 0.0f
    };

    uint16_t indices[] = {
        0, 1, 2
    };

    auto hardwareVertexBuffer = hardwareResourceManager.createVertexBuffer("triangle vertex buffer", 3, 3 * sizeof(float), vertices);
    auto hardwareIndexBuffer = hardwareResourceManager.createIndexBuffer("triangle index buffer", 3, Syrinx::IndexType::UINT16, indices);
    auto vertexInputState = hardwareResourceManager.createVertexInputState("triangle vertex input state");

    Syrinx::VertexAttributeDescription positionAttributeDesc;
    positionAttributeDesc.setSemantic(Syrinx::VertexAttributeSemantic::Position)
                         .setLocation(0)
                         .setBindingPoint(0)
                         .setDataOffset(0)
                         .setDataType(Syrinx::VertexAttributeDataType::FLOAT3);

    Syrinx::VertexAttributeLayoutDesc vertexAttributeLayoutDesc;
    vertexAttributeLayoutDesc.addVertexAttributeDesc(positionAttributeDesc);

    vertexInputState->setVertexAttributeLayoutDesc(std::move(vertexAttributeLayoutDesc));
    vertexInputState->setVertexBuffer(0, hardwareVertexBuffer);
    vertexInputState->setIndexBuffer(hardwareIndexBuffer);
    vertexInputState->setup();

    while (renderWindow->isOpen()) {
        float defaultValueForColorAttachment[] = {0.5, 0.0, 0.5, 1.0};
        glClearNamedFramebufferfv(0, GL_COLOR, 0, defaultValueForColorAttachment);
        float defaultDepthValue = 1.0;
        glClearNamedFramebufferfv(0, GL_DEPTH, 0, &defaultDepthValue);

        glBindVertexArray(vertexInputState->getHandle());
        glBindProgramPipeline(programPipeline->getHandle());
        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, nullptr);

        renderWindow->swapBuffer();
    }

    return 0;
}