#include <Image/SyrinxImage.h>
#include <Image/SyrinxImageReader.h>
#include <Logging/SyrinxLogManager.h>
#include <HardwareResource/SyrinxProgramStage.h>
#include <HardwareResource/SyrinxProgramPipeline.h>
#include <HardwareResource/SyrinxHardwareVertexBuffer.h>
#include <HardwareResource/SyrinxHardwareIndexBuffer.h>
#include <HardwareResource/SyrinxVertexInputState.h>
#include <HardwareResource/SyrinxHardwareTexture.h>
#include <ResourceManager/SyrinxFileManager.h>
#include <RenderPipeline/SyrinxDisplayDevice.h>


int main(int argc, char *argv[])
{
    Syrinx::LogManager *logManager = new Syrinx::LogManager();
    auto fileManager = new Syrinx::FileManager();
    Syrinx::DisplayDevice displayDevice;
    displayDevice.setMajorVersionNumber(4);
    displayDevice.setMinorVersionNumber(5);
    displayDevice.setDebugMessageHandler(Syrinx::DefaultDebugHandler);

    const int WINDOW_WIDTH = 512;
    const int WINDOW_HEIGHT = 512;
    displayDevice.createWindow("Texture MipMap Sample", WINDOW_WIDTH, WINDOW_HEIGHT);
    auto renderWindow = displayDevice.getRenderWindow();

    fileManager->addSearchPath("../../Medias/");
    auto loadProgramSource = [fileManager](const std::string& fileName) {
        auto fileStream = fileManager->openFile(fileName, Syrinx::FileAccessMode::READ);
        return fileStream->getAsString();
    };


    const std::string textureCopyVertexShaderSource = loadProgramSource("ImagePass_VS.glsl");
    const std::string textureCopyFragmentShaderSource = loadProgramSource("TextureCopy_FS.glsl");

    auto textureCopyVertexProgram = std::make_shared<Syrinx::ProgramStage>("texture copy vertex program");
    textureCopyVertexProgram->setType(Syrinx::ProgramStageType::VertexStage);
    textureCopyVertexProgram->setSource(textureCopyVertexShaderSource);
    textureCopyVertexProgram->create();

    auto textureCopyFragmentProgram = std::make_shared<Syrinx::ProgramStage>("texture copy fragment program");
    textureCopyFragmentProgram->setType(Syrinx::ProgramStageType::FragmentStage);
    textureCopyFragmentProgram->setSource(textureCopyFragmentShaderSource);
    textureCopyFragmentProgram->create();

    Syrinx::ProgramPipeline textureCopyProgramPipeline("texture copy program pipeline");
    textureCopyProgramPipeline.create();
    textureCopyProgramPipeline.bindProgramStage(textureCopyVertexProgram.get());
    textureCopyProgramPipeline.bindProgramStage(textureCopyFragmentProgram.get());

    const std::string imageFile = "mipmap-test.png";
    auto [imageExist, imageFilePath] = fileManager->findFile(imageFile);
    if (!imageExist) {
        SYRINX_FAULT_FMT("can not find image [{}]", imageFile);
    }

    Syrinx::ImageReader imageReader;
    Syrinx::Image image = imageReader.read(imageFilePath, Syrinx::ImageFormat::RGB8);
    Syrinx::TextureSamplingSetting textureSamplingSetting;
    textureSamplingSetting.setMagFilterMethod(Syrinx::TextureMagFilterMethod::NEAREST);
    textureSamplingSetting.setMinFilterMethod(Syrinx::TextureMinFilterMethod::NEAREST_MIPMAP_NEAREST);
    textureSamplingSetting.setWrapSMethod(Syrinx::TextureWrapMethod::CLAMP_TO_EDGE);
    textureSamplingSetting.setWrapTMethod(Syrinx::TextureWrapMethod::CLAMP_TO_EDGE);

    Syrinx::HardwareTexture hardwareTexture("quad texture");
    hardwareTexture.setSamplingSetting(textureSamplingSetting);
    hardwareTexture.setType(Syrinx::TextureType::TEXTURE_2D);
    hardwareTexture.setPixelFormat(Syrinx::PixelFormat::RGB8);
    hardwareTexture.setWidth(image.getWidth());
    hardwareTexture.setHeight(image.getHeight());
    hardwareTexture.create();
    hardwareTexture.write(image.getData(), image.getWidth(), image.getHeight());
    hardwareTexture.generateTextureMipMap();


    float quadVertices[] = {
            1.0f, 1.0f, 0.0f,   1.0f, 1.0f,
            -1.0f, 1.0f, 0.0f,  0.0f, 1.0f,
            1.0f,-1.0f, 0.0f,   1.0f, 0.0f,
            -1.0f,-1.0f, 0.0f,  0.0f, 0.0f
    };

    uint16_t quadIndices[] = {
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
    quadIndexBuffer.setIndexType(Syrinx::IndexType::UINT16);
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

    while (renderWindow->isOpen()) {
        float defaultValueForColorAttachment[] = {1.0, 1.0, 0.0, 1.0};
        glClearNamedFramebufferfv(0, GL_COLOR, 0, defaultValueForColorAttachment);
        float defaultDepthValue = 1.0;
        glClearNamedFramebufferfv(0, GL_DEPTH, 0, &defaultDepthValue);

        glBindProgramPipeline(textureCopyProgramPipeline.getHandle());
        glBindVertexArray(quadVertexInputState.getHandle());
        glBindTextureUnit(0, hardwareTexture.getHandle());
        textureCopyFragmentProgram->updateParameter("uTexSampler", 0);
        textureCopyFragmentProgram->updateParameter("uMipMapLevel", hardwareTexture.getMaxMipMapLevel());
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);
        glBindVertexArray(0);
        
        renderWindow->swapBuffer();
    }

    return 0;
}