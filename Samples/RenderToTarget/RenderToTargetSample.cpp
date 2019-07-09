#include <FileSystem/SyrinxFileManager.h>
#include <Graphics/Program/SyrinxProgramCompiler.h>
#include <Logging/SyrinxLogManager.h>
#include <Manager/SyrinxHardwareResourceManager.h>
#include <HardwareResource/SyrinxProgramStage.h>
#include <HardwareResource/SyrinxProgramPipeline.h>
#include <HardwareResource/SyrinxVertexInputState.h>
#include <Pipeline/SyrinxDisplayDevice.h>
#include <Pipeline/SyrinxEngineSetting.h>
#include <HardwareResource/SyrinxDepthTexture.h>
#include <HardwareResource/SyrinxRenderTarget.h>
#include <HardwareResource/SyrinxRenderTexture.h>


int main(int argc, char *argv[])
{
    auto logManager = std::make_unique<Syrinx::LogManager>();
    Syrinx::DisplayDevice displayDevice;
    displayDevice.setMajorVersionNumber(4);
    displayDevice.setMinorVersionNumber(6);
    displayDevice.setDebugMessageHandler(Syrinx::DefaultDebugHandler);
    displayDevice.createWindow("Render To Target Sample", 800, 600);
    auto renderWindow = displayDevice.getRenderWindow();

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
        "   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
        "}\n";

    const std::string quadVertexProgramSource =
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec2 texCoord;\n"
        "layout(location = 0) out vec2 TexCoord;\n"
        "out gl_PerVertex {\n"
        "    vec4 gl_Position;\n"
        "};\n"
        "void main(){\n"
        "    gl_Position=vec4(position,1.0f);\n"
        "    TexCoord=vec2(texCoord.x,texCoord.y);\n"
        "}\n";

    const std::string quadFragmentProgramSource =
        "layout(location = 0) in vec2 TexCoord;\n"
        "layout(location = 0) out vec4 color;\n"
        "uniform sampler2D ourTexture1;\n"
        "void main(){\n"
        "    color = vec4(texture(ourTexture1,TexCoord).rgb,1.0f);\n"
        "}\n";


    Syrinx::ProgramCompiler compiler;
    Syrinx::HardwareResourceManager hardwareResourceManager;

    auto vertexProgramBinarySource = compiler.compile("vertex program", vertexProgramSource, Syrinx::ProgramStageType::VertexStage);
    auto vertexProgram = hardwareResourceManager.createProgramStage("vertex program", std::move(vertexProgramBinarySource) , Syrinx::ProgramStageType::VertexStage);
    auto fragmentProgramBinarySource = compiler.compile("fragment program", fragmentProgramSource, Syrinx::ProgramStageType::FragmentStage);
    auto fragmentProgram = hardwareResourceManager.createProgramStage("fragment program", std::move(fragmentProgramBinarySource), Syrinx::ProgramStageType::FragmentStage);
    auto drawTriangleProgramPipeline = hardwareResourceManager.createProgramPipeline("draw triangle");
    drawTriangleProgramPipeline->bindProgramStage(vertexProgram);
    drawTriangleProgramPipeline->bindProgramStage(fragmentProgram);

    auto quadVertexProgramBinarySource = compiler.compile("quad vertex program", quadVertexProgramSource, Syrinx::ProgramStageType::VertexStage);
    auto quadVertexProgram = hardwareResourceManager.createProgramStage("quad vertex program", std::move(quadVertexProgramBinarySource) , Syrinx::ProgramStageType::VertexStage);
    auto quadFragmentProgramBinarySource = compiler.compile("quad fragment program", quadFragmentProgramSource, Syrinx::ProgramStageType::FragmentStage);
    auto quadFragmentProgram = hardwareResourceManager.createProgramStage("quad fragment program", std::move(quadFragmentProgramBinarySource), Syrinx::ProgramStageType::FragmentStage);
    auto drawQuadProgramPipeline = hardwareResourceManager.createProgramPipeline("draw quad program pipeline");
    drawQuadProgramPipeline->bindProgramStage(quadVertexProgram);
    drawQuadProgramPipeline->bindProgramStage(quadFragmentProgram);

    float triangleVertices[] = {
        -0.5f, -0.5f, 0.0f,
        0.5f, -0.5f, 0.0f,
        0.0f,  0.5f, 0.0f
    };

    uint32_t triangleIndices[] = {
        0, 1, 2
    };

    auto triangleVertexBuffer = hardwareResourceManager.createVertexBuffer("triangle vertex buffer", 3, 3 * sizeof(float), triangleVertices);
    auto triangleIndexBuffer = hardwareResourceManager.createIndexBuffer("triangle index buffer", 3, Syrinx::IndexType::UINT32, triangleIndices);

    Syrinx::VertexAttributeDescription triangleVertexAttributeDescription(0, Syrinx::VertexAttributeSemantic::Position, Syrinx::VertexAttributeDataType::FLOAT3);
    Syrinx::VertexDataDescription triangleDataDescription(triangleVertexBuffer, 0, 0, 3 * sizeof(float));
    Syrinx::VertexInputState triangleInputState("triangle input state");
    triangleInputState.addVertexAttributeDescription(triangleVertexAttributeDescription);
    triangleInputState.addVertexDataDescription(triangleDataDescription);
    triangleInputState.addIndexBuffer(triangleIndexBuffer);
    triangleInputState.create();


    float quadVertices[] = {
        0.5f, 0.5f, 0.0f,    1.0f, 1.0f,
        -0.5f, 0.5f, 0.0f,    0.0f, 1.0f,
        0.5f,-0.5f, 0.0f,    1.0f, 0.0f,
        -0.5f,-0.5f, 0.0f,    0.0f, 0.0f
    };

    uint32_t quadIndices[] = {
        0, 1, 2,
        2, 1, 3
    };

    auto quadVertexBuffer = hardwareResourceManager.createVertexBuffer("quad vertex buffer", 4, 5 * sizeof(float), quadVertices);
    auto quadIndexBuffer = hardwareResourceManager.createIndexBuffer("quad index buffer", 6, Syrinx::IndexType::UINT32, quadIndices);
    Syrinx::VertexAttributeDescription quadPositionAttributeDescription(0, Syrinx::VertexAttributeSemantic::Position, Syrinx::VertexAttributeDataType::FLOAT3);
    Syrinx::VertexAttributeDescription quadTexCoordAttributeDescription(1, Syrinx::VertexAttributeSemantic::TexCoord, Syrinx::VertexAttributeDataType::FLOAT2);
    Syrinx::VertexDataDescription quadPositionDataDescription(quadVertexBuffer, 0, 0, 5 * sizeof(float));
    Syrinx::VertexDataDescription quadTexCoordDataDescription(quadVertexBuffer, 1, 3 * sizeof(float), 5 * sizeof(float));
    Syrinx::VertexInputState quadVertexInputState("quad input state");
    quadVertexInputState.addVertexAttributeDescription(quadPositionAttributeDescription);
    quadVertexInputState.addVertexAttributeDescription(quadTexCoordAttributeDescription);
    quadVertexInputState.addVertexDataDescription(quadPositionDataDescription);
    quadVertexInputState.addVertexDataDescription(quadTexCoordDataDescription);
    quadVertexInputState.addIndexBuffer(quadIndexBuffer);
    quadVertexInputState.create();

    Syrinx::RenderTarget::Desc desc;
    desc.setColorAttachment(0, Syrinx::PixelFormat::RGBAF)
        .setDepthStencilAttachment(Syrinx::PixelFormat::DEPTH32F);
    auto renderTarget = hardwareResourceManager.createRenderTarget("render target", desc, renderWindow->getWidth(), renderWindow->getHeight());
    SYRINX_ASSERT(renderTarget);

    while (renderWindow->isOpen()) {
        renderTarget->clearRenderTexture();
        renderTarget->clearDepthTexture();
        glEnable(GL_DEPTH_TEST);
        glBindFramebuffer(GL_FRAMEBUFFER, renderTarget->getHandle());
        glBindVertexArray(triangleInputState.getHandle());
        glBindProgramPipeline(drawTriangleProgramPipeline->getHandle());
        glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, nullptr);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        float quadDefaultValueForColorAttachment[] = {0.0, 1.0, 0.0, 1.0};
        float defaultDepthValue = 1.0;
        glClearNamedFramebufferfv(0, GL_COLOR, 0, quadDefaultValueForColorAttachment);
        glClearNamedFramebufferfv(0, GL_DEPTH, 0, &defaultDepthValue);
        glBindVertexArray(quadVertexInputState.getHandle());
        glBindProgramPipeline(drawQuadProgramPipeline->getHandle());

        auto colorAttachment = renderTarget->getColorAttachment(0);
        SYRINX_ASSERT(colorAttachment);
        glBindTextureUnit(0, colorAttachment->getHandle());
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

        renderWindow->swapBuffer();
    }

    return 0;
}