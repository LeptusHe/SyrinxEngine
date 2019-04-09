#include <Logging/SyrinxLogManager.h>
#include <HardwareResource/SyrinxProgramStage.h>
#include <HardwareResource/SyrinxProgramPipeline.h>
#include <HardwareResource/SyrinxVertexInputState.h>
#include <RenderResource/SyrinxRenderTexture.h>
#include <Pipeline/SyrinxDisplayDevice.h>
#include <Pipeline/SyrinxEngineSetting.h>
#include <RenderResource/SyrinxDepthTexture.h>
#include <RenderResource/SyrinxRenderTarget.h>


int main(int argc, char *argv[])
{
    Syrinx::LogManager *logManager = new Syrinx::LogManager();
    Syrinx::DisplayDevice displayDevice;
    displayDevice.setMajorVersionNumber(4);
    displayDevice.setMinorVersionNumber(5);
    displayDevice.setDebugMessageHandler(Syrinx::DefaultDebugHandler);
    displayDevice.createWindow("RenderToTarget", 800, 600);
    auto renderWindow = displayDevice.getRenderWindow();


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
        "layout(location = 0) out vec4 FragColor;\n"
        "void main()\n"
        "{\n"
        "   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
        "}\n";

    const std::string quadVertexProgramSource =
        "#version 450 core\n"
        "layout (location=0) in vec3 position;\n"
        "layout (location=1) in vec2 texCoord;\n"
        "out gl_PerVertex {\n"
        "    vec4 gl_Position;\n"
        "};\n"
        "out vec2 TexCoord;\n"
        "void main(){\n"
        "gl_Position=vec4(position,1.0f);\n"
        "TexCoord=vec2(texCoord.x,texCoord.y);\n"
        "}\n";

    const std::string quadFragmentProgramSource =
        "#version 450 core\n"
        "in vec2 TexCoord;\n"
        "out vec4 color;\n"
        "uniform sampler2D ourTexture1;\n"
        "void main(){\n"
        "    color = vec4(texture(ourTexture1,TexCoord).rgb,1.0f);\n"
        "}\n";


    auto vertexProgram = std::make_unique<Syrinx::ProgramStage>("vertex program");
    vertexProgram->setType(Syrinx::ProgramStageType::VertexStage);
    vertexProgram->setSource(vertexProgramSource);
    vertexProgram->create();

    auto fragmentProgram = std::make_unique<Syrinx::ProgramStage>("fragment program");
    fragmentProgram->setType(Syrinx::ProgramStageType::FragmentStage);
    fragmentProgram->setSource(fragmentProgramSource);
    fragmentProgram->create();

    auto drawTriangleProgramPipeline = std::make_unique<Syrinx::ProgramPipeline>("draw triangle");
    drawTriangleProgramPipeline->create();
    drawTriangleProgramPipeline->bindProgramStage(vertexProgram.get());
    drawTriangleProgramPipeline->bindProgramStage(fragmentProgram.get());

    auto quadVertexProgram = std::make_unique<Syrinx::ProgramStage>("quad vertex program");
    quadVertexProgram->setType(Syrinx::ProgramStageType::VertexStage);
    quadVertexProgram->setSource(quadVertexProgramSource);
    quadVertexProgram->create();

    auto quadFragmentProgram = std::make_unique<Syrinx::ProgramStage>("quad fragment program");
    quadFragmentProgram->setType(Syrinx::ProgramStageType::FragmentStage);
    quadFragmentProgram->setSource(quadFragmentProgramSource);
    quadFragmentProgram->create();

    auto drawQuadProgramPipeline = std::make_unique<Syrinx::ProgramPipeline>("draw quad");
    drawQuadProgramPipeline->create();
    drawQuadProgramPipeline->bindProgramStage(quadVertexProgram.get());
    drawQuadProgramPipeline->bindProgramStage(quadFragmentProgram.get());


    float triangleVertices[] = {
        -0.5f, -0.5f, 0.0f,
        0.5f, -0.5f, 0.0f,
        0.0f,  0.5f, 0.0f
    };

    uint32_t triangleIndices[] = {
        0, 1, 2
    };

    auto triangleHardwareVertexBuffer = std::make_unique<Syrinx::HardwareBuffer>("triangle vertex buffer");
    Syrinx::HardwareVertexBuffer triangleVertexBuffer(std::move(triangleHardwareVertexBuffer));
    triangleVertexBuffer.setVertexNumber(3);
    triangleVertexBuffer.setVertexSizeInBytes(3 * sizeof(float));
    triangleVertexBuffer.setData(triangleVertices);
    triangleVertexBuffer.create();

    auto triangleHardwareIndexBuffer = std::make_unique<Syrinx::HardwareBuffer>("triangle index buffer");
    Syrinx::HardwareIndexBuffer triangleIndexBuffer(std::move(triangleHardwareIndexBuffer));
    triangleIndexBuffer.setIndexType(Syrinx::IndexType::UINT32);
    triangleIndexBuffer.setIndexNumber(3);
    triangleIndexBuffer.setData(triangleIndices);
    triangleIndexBuffer.create();

    Syrinx::VertexAttributeDescription triangleVertexAttributeDescription(0, Syrinx::VertexAttributeSemantic::Position, Syrinx::VertexAttributeDataType::FLOAT3);
    Syrinx::VertexDataDescription triangleDataDescription(&triangleVertexBuffer, 0, 0, 3 * sizeof(float));

    Syrinx::VertexInputState triangleInputState("triangle input state");
    triangleInputState.addVertexAttributeDescription(triangleVertexAttributeDescription);
    triangleInputState.addVertexDataDescription(triangleDataDescription);
    triangleInputState.addIndexBuffer(&triangleIndexBuffer);
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

    auto quadHardwareVertexBuffer = std::make_unique<Syrinx::HardwareBuffer>("quad vertex buffer");
    Syrinx::HardwareVertexBuffer quadVertexBuffer(std::move(quadHardwareVertexBuffer));
    quadVertexBuffer.setVertexNumber(4);
    quadVertexBuffer.setVertexSizeInBytes(5 * sizeof(float));
    quadVertexBuffer.setData(quadVertices);
    quadVertexBuffer.create();

    auto quadHardwareIndexBuffer = std::make_unique<Syrinx::HardwareBuffer>("quad index buffer");
    Syrinx::HardwareIndexBuffer quadIndexBuffer(std::move(quadHardwareIndexBuffer));
    quadIndexBuffer.setIndexType(Syrinx::IndexType::UINT32);
    quadIndexBuffer.setIndexNumber(6);
    quadIndexBuffer.setData(quadIndices);
    quadIndexBuffer.create();

    Syrinx::VertexAttributeDescription quadPositionAttributeDescription(0, Syrinx::VertexAttributeSemantic::Position, Syrinx::VertexAttributeDataType::FLOAT3);
    Syrinx::VertexAttributeDescription quadTexCoordAttributeDescription(1, Syrinx::VertexAttributeSemantic::TexCoord, Syrinx::VertexAttributeDataType::FLOAT2);
    Syrinx::VertexDataDescription quadPositionDataDescription(&quadVertexBuffer, 0, 0, 5 * sizeof(float));
    Syrinx::VertexDataDescription quadTexCoordDataDescription(&quadVertexBuffer, 1, 3 * sizeof(float), 5 * sizeof(float));

    Syrinx::VertexInputState quadVertexInputState("quad input state");
    quadVertexInputState.addVertexAttributeDescription(quadPositionAttributeDescription);
    quadVertexInputState.addVertexAttributeDescription(quadTexCoordAttributeDescription);
    quadVertexInputState.addVertexDataDescription(quadPositionDataDescription);
    quadVertexInputState.addVertexDataDescription(quadTexCoordDataDescription);
    quadVertexInputState.addIndexBuffer(&quadIndexBuffer);
    quadVertexInputState.create();


    Syrinx::HardwareTexture colorTexture("color texture");
    colorTexture.setType(Syrinx::TextureType::TEXTURE_2D);
    colorTexture.setPixelFormat(Syrinx::PixelFormat::RGBA8);
    colorTexture.setWidth(renderWindow->getWidth());
    colorTexture.setHeight(renderWindow->getHeight());
    colorTexture.create();

    Syrinx::HardwareTexture depthTexture("depth texture");
    depthTexture.setType(Syrinx::TextureType::TEXTURE_2D);
    depthTexture.setPixelFormat(Syrinx::PixelFormat::DEPTH24);
    depthTexture.setWidth(renderWindow->getWidth());
    depthTexture.setHeight(renderWindow->getHeight());
    depthTexture.create();

    Syrinx::RenderTexture colorRenderTexture("color render texture", &colorTexture);
    Syrinx::RenderTexture depthRenderTexture("depth render texture", &depthTexture);
    Syrinx::DepthTexture depthAttachment("depth attachment", &depthRenderTexture);

    Syrinx::RenderTarget renderTarget("render target");
    renderTarget.addRenderTexture(0, &colorRenderTexture);
    renderTarget.addDepthTexture(&depthAttachment);
    renderTarget.create();

    while (renderWindow->isOpen()) {
        renderTarget.clearRenderTexture();
        renderTarget.clearDepthTexture();
        glEnable(GL_DEPTH_TEST);
        glBindFramebuffer(GL_FRAMEBUFFER, renderTarget.getHandle());
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
        glBindTextureUnit(0, colorTexture.getHandle());
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

        renderWindow->swapBuffer();
    }

    return 0;
}