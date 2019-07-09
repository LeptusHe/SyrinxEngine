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
    Syrinx::LogManager *logManager = new Syrinx::LogManager();
    Syrinx::DisplayDevice displayDevice;

    displayDevice.setMajorVersionNumber(4);
    displayDevice.setMinorVersionNumber(6);
    displayDevice.setDebugMessageHandler(Syrinx::DefaultDebugHandler);
    auto renderWindow = displayDevice.createWindow("Update Uniform Sample", 800, 600);

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
        "layout(std140, binding = 0) uniform material {\n"
        "    vec4 diffuseColor;\n"
        "};\n"
        "void main()\n"
        "{\n"
        "   FragColor = vec4(diffuseColor.xyz, 1.0f);\n"
        "}\n";

    Syrinx::ProgramCompiler compiler;
    Syrinx::HardwareResourceManager hardwareResourceManager;
    auto vertexProgramBinarySource = compiler.compile("vertex", vertexProgramSource, Syrinx::ProgramStageType::VertexStage);
    auto vertexProgram = hardwareResourceManager.createProgramStage("update uniform vertex program",
                                                                    std::move(vertexProgramBinarySource),
                                                                    Syrinx::ProgramStageType::VertexStage);

    auto fragmentProgramBinarySource = compiler.compile("fragment", fragmentProgramSource, Syrinx::ProgramStageType::FragmentStage);
    auto fragmentProgram = hardwareResourceManager.createProgramStage("update uniform fragment program",
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

    auto vertexBuffer = std::make_unique<Syrinx::HardwareBuffer>("triangle vertex buffer");
    Syrinx::HardwareVertexBuffer hardwareVertexBuffer(std::move(vertexBuffer));
    hardwareVertexBuffer.setVertexNumber(3);
    hardwareVertexBuffer.setVertexSizeInBytes(3 * sizeof(float));
    hardwareVertexBuffer.initData(vertices);
    hardwareVertexBuffer.create();

    auto indexBuffer = std::make_unique<Syrinx::HardwareBuffer>("triangle index buffer");
    Syrinx::HardwareIndexBuffer hardwareIndexBuffer(std::move(indexBuffer));
    hardwareIndexBuffer.setIndexType(Syrinx::IndexType::UINT16);
    hardwareIndexBuffer.setIndexNumber(3);
    hardwareIndexBuffer.initData(indices);
    hardwareIndexBuffer.create();

    Syrinx::VertexAttributeDescription vertexAttributeDescription(0, Syrinx::VertexAttributeSemantic::Position, Syrinx::VertexAttributeDataType::FLOAT3);
    Syrinx::VertexDataDescription vertexDataDescription(&hardwareVertexBuffer, 0, 0, 3 * sizeof(float));

    Syrinx::VertexInputState vertexInputState("triangle vertex input state");
    vertexInputState.addVertexAttributeDescription(vertexAttributeDescription);
    vertexInputState.addVertexDataDescription(vertexDataDescription);
    vertexInputState.addIndexBuffer(&hardwareIndexBuffer);
    vertexInputState.create();

    auto generateColor = [](unsigned int counter) -> std::tuple<float, float, float> {
        unsigned int colorIndex = (counter / 10) % 3;
        switch (colorIndex) {
            case 0: return {1.0, 0.0, 0.0};
            case 1: return {0.0, 1.0, 0.0};
            case 2: return {0.0, 0.0, 1.0};
            default: SYRINX_ASSERT(false && "invalid color index");
        }
        return {0.0, 0.0, 0.0};
    };

    auto& fragmentProgramVars = *(fragmentProgram->getProgramVars());
    unsigned int frameCounter = 0;
    while (renderWindow->isOpen()) {
        frameCounter += 1;
        float defaultValueForColorAttachment[] = {0.5, 0.0, 0.5, 1.0};
        glClearNamedFramebufferfv(0, GL_COLOR, 0, defaultValueForColorAttachment);
        float defaultDepthValue = 1.0;
        glClearNamedFramebufferfv(0, GL_DEPTH, 0, &defaultDepthValue);

        glBindVertexArray(vertexInputState.getHandle());
        glBindProgramPipeline(programPipeline->getHandle());

        auto [red, green, blue] = generateColor(frameCounter);
        auto& materialUniformBlock = fragmentProgramVars["material"];
        materialUniformBlock["diffuseColor"] = glm::vec4(red, green, blue, 1.0);
        materialUniformBlock.uniformBuffer->updateToGPU();

        auto uniformBufferHandle = materialUniformBlock.uniformBuffer->getHandle();
        glBindBufferBase(GL_UNIFORM_BUFFER, materialUniformBlock.binding, uniformBufferHandle);

        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, nullptr);
        renderWindow->swapBuffer();
    }

    return 0;
}
