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
    auto renderWindow = displayDevice.createWindow("Load Texture Sample", 800, 600);

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
        "out vec4 FragColor;\n"
        "uniform sampler2D uDiffuseTex;\n"
        "in vec2 texCoord;\n"
        "void main()\n"
        "{\n"
        "   FragColor = texture(uDiffuseTex, texCoord);\n"
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
    const std::string imageFile = "checkerboard.png";
    auto [imageExist, filePath] = fileManager.findFile(imageFile);
    if (!imageExist) {
        SYRINX_FAULT_FMT("can not find image [{}]", imageFile);
        return 0;
    }

    Syrinx::ImageReader imageReader;
    Syrinx::Image image = imageReader.read(filePath, Syrinx::ImageFormat::RGBA8);
    Syrinx::HardwareTexture hardwareTexture("quad texture");
    hardwareTexture.setType(Syrinx::TextureType::TEXTURE_2D);
    hardwareTexture.setPixelFormat(Syrinx::PixelFormat::RGBA8);
    hardwareTexture.setWidth(image.getWidth());
    hardwareTexture.setHeight(image.getHeight());
    hardwareTexture.create();
    hardwareTexture.write(image.getData(), image.getWidth(), image.getHeight());

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

    while (renderWindow->isOpen()) {
        float defaultValueForColorAttachment[] = {1.0, 1.0, 0.0, 1.0};
        glClearNamedFramebufferfv(0, GL_COLOR, 0, defaultValueForColorAttachment);
        float defaultDepthValue = 1.0;
        glClearNamedFramebufferfv(0, GL_DEPTH, 0, &defaultDepthValue);

        glBindTextureUnit(0, hardwareTexture.getHandle());
        fragmentProgram->updateParameter("uDiffuseTex", 0);

        glBindVertexArray(vertexInputState.getHandle());
        glBindProgramPipeline(programPipeline.getHandle());
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);

        renderWindow->swapBuffer();
    }

    return 0;
}
