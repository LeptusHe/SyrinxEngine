#include <Image/SyrinxImage.h>
#include <Image/SyrinxImageReader.h>
#include <Logging/SyrinxLogManager.h>
#include <RenderPipeline/SyrinxDisplayDevice.h>
#include <HardwareResource/SyrinxProgramStage.h>
#include <HardwareResource/SyrinxProgramPipeline.h>
#include <HardwareResource/SyrinxHardwareVertexBuffer.h>
#include <HardwareResource/SyrinxHardwareIndexBuffer.h>
#include <HardwareResource/SyrinxVertexInputState.h>
#include <HardwareResource/SyrinxHardwareTexture.h>
#include <ResourceManager/SyrinxFileManager.h>


int main(int argc, char *argv[])
{
    Syrinx::LogManager *logManager = new Syrinx::LogManager();
    Syrinx::DisplayDevice displayDevice;
    displayDevice.setMajorVersionNumber(4);
    displayDevice.setMinorVersionNumber(5);
    displayDevice.setDebugMessageHandler(Syrinx::DefaultDebugHandler);
    auto renderWindow = displayDevice.createWindow("Bindless Texture Sample", 800, 600);

    const std::string vertexProgramSource =
            "#version 450 core\n"
            "layout(location = 0) in vec3 aPos;\n"
            "out gl_PerVertex {\n"
            "    vec4 gl_Position;\n"
            "};\n"
            "out vec2 texCoord;\n"
            "void main() {\n"
            "    texCoord = aPos.xy + vec2(0.5); \n"
            "    gl_Position = vec4(aPos, 1.0);\n"
            "}\n";

    const std::string fragmentProgramSource =
            "#version 450 core\n"
            "#extension GL_NV_bindless_texture : require\n"
            "#extension GL_NV_gpu_shader5 : require\n"
            "out vec4 FragColor;\n"
            "layout (std140, binding = 0) uniform TextureHandles {\n"
            "   sampler2D texture1;\n"
            "   sampler2D texture2;\n"
            "};\n"
            "in vec2 texCoord;\n"
            "void main()\n"
            "{\n"
            "   FragColor = vec4(1.0, 0.0, 0.0, 1.0);\n"
            "   if (gl_FragCoord.x <= 400) {\n"
            "       FragColor = texture(texture2, texCoord);\n"
            "   } else {\n"
            "       FragColor = texture(texture1, texCoord);\n"
            "   }\n"
            "}\n";

    auto vertexProgram = std::make_unique<Syrinx::ProgramStage>("load texture vertex program");
    vertexProgram->setType(Syrinx::ProgramStageType::VertexStage);
    vertexProgram->setSource(vertexProgramSource);
    vertexProgram->create();

    auto fragmentProgram = std::make_unique<Syrinx::ProgramStage>("load texture fragment program");
    fragmentProgram->setType(Syrinx::ProgramStageType::FragmentStage);
    fragmentProgram->setSource(fragmentProgramSource);
    fragmentProgram->create();

    Syrinx::ProgramPipeline programPipeline("load texture program pipeline");
    programPipeline.create();
    programPipeline.bindProgramStage(vertexProgram.get());
    programPipeline.bindProgramStage(fragmentProgram.get());

    Syrinx::FileManager fileManager;
    fileManager.addSearchPath("../../Medias");

    Syrinx::ImageReader imageReader;
    Syrinx::Image image1 = imageReader.read("../../Medias/Textures/checkerboard.png", Syrinx::ImageFormat::RGBA8);
    Syrinx::HardwareTexture hardwareTexture1("quad texture");
    hardwareTexture1.setType(Syrinx::TextureType::TEXTURE_2D);
    hardwareTexture1.setPixelFormat(Syrinx::PixelFormat::RGBA8);
    hardwareTexture1.setWidth(image1.getWidth());
    hardwareTexture1.setHeight(image1.getHeight());
    hardwareTexture1.create();
    hardwareTexture1.write(image1.getData(), image1.getWidth(), image1.getHeight());

    Syrinx::Image image2 = imageReader.read("../../Medias/Textures/mipmap-test.png", Syrinx::ImageFormat::RGB8);
    Syrinx::HardwareTexture hardwareTexture2("quad texture");
    hardwareTexture2.setType(Syrinx::TextureType::TEXTURE_2D);
    hardwareTexture2.setPixelFormat(Syrinx::PixelFormat::RGB8);
    hardwareTexture2.setWidth(image2.getWidth());
    hardwareTexture2.setHeight(image2.getHeight());
    hardwareTexture2.create();
    hardwareTexture2.write(image2.getData(), image2.getWidth(), image2.getHeight());

    GLuint64 textureHandle1 = glGetTextureHandleARB(hardwareTexture1.getHandle());
    glMakeTextureHandleResidentARB(textureHandle1);
    GLuint64 textureHandle2 = glGetTextureHandleARB(hardwareTexture2.getHandle());
    glMakeTextureHandleResidentARB(textureHandle2);

    GLuint64 textureHandles[] = {
            textureHandle1,
            textureHandle2
    };

    Syrinx::HardwareBuffer uniformBuffer("texture handle uniform buffer");
    uniformBuffer.setSize(sizeof(textureHandles));
    uniformBuffer.setData(textureHandles);
    uniformBuffer.create();
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, uniformBuffer.getHandle());

    float vertices[] = {
            -0.5f, -0.5f, 0.0f,
            0.5f, -0.5f, 0.0f,
            -0.5f, 0.5f, 0.0f,
            0.5f,  0.5f, 0.0f
    };

    uint16_t indices[] = {
            0, 1, 2,
            1, 2, 3
    };

    auto vertexBuffer = std::make_unique<Syrinx::HardwareBuffer>("quad vertex buffer");
    Syrinx::HardwareVertexBuffer hardwareVertexBuffer(std::move(vertexBuffer));
    hardwareVertexBuffer.setVertexNumber(4);
    hardwareVertexBuffer.setVertexSizeInBytes(3 * sizeof(float));
    hardwareVertexBuffer.setData(vertices);
    hardwareVertexBuffer.create();

    auto indexBuffer = std::make_unique<Syrinx::HardwareBuffer>("quad index buffer");
    Syrinx::HardwareIndexBuffer hardwareIndexBuffer(std::move(indexBuffer));
    hardwareIndexBuffer.setIndexType(Syrinx::IndexType::UINT16);
    hardwareIndexBuffer.setIndexNumber(6);
    hardwareIndexBuffer.setData(indices);
    hardwareIndexBuffer.create();

    Syrinx::VertexAttributeDescription vertexAttributeDescription(0, Syrinx::VertexAttributeSemantic::Position, Syrinx::VertexAttributeDataType::FLOAT3);
    Syrinx::VertexDataDescription vertexDataDescription(&hardwareVertexBuffer, 0, 0, 3 * sizeof(float));

    Syrinx::VertexInputState vertexInputState("quad vertex input state");
    vertexInputState.addVertexAttributeDescription(vertexAttributeDescription);
    vertexInputState.addVertexDataDescription(vertexDataDescription);
    vertexInputState.addIndexBuffer(&hardwareIndexBuffer);
    vertexInputState.create();


    while (renderWindow->isOpen()) {
        float defaultValueForColorAttachment[] = {1.0, 1.0, 0.0, 1.0};
        glClearNamedFramebufferfv(0, GL_COLOR, 0, defaultValueForColorAttachment);
        float defaultDepthValue = 1.0;
        glClearNamedFramebufferfv(0, GL_DEPTH, 0, &defaultDepthValue);

        glBindVertexArray(vertexInputState.getHandle());
        glBindProgramPipeline(programPipeline.getHandle());
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);

        renderWindow->swapBuffer();
    }

    return 0;
}
