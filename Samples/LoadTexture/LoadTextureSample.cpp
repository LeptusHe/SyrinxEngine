#include <Image/SyrinxImage.h>
#include <Image/SyrinxImageReader.h>
#include <FileSystem/SyrinxFileManager.h>
#include <Logging/SyrinxLogManager.h>
#include <Program/SyrinxProgramCompiler.h>
#include <Manager/SyrinxHardwareResourceManager.h>
#include <HardwareResource/SyrinxProgramStage.h>
#include <HardwareResource/SyrinxProgramPipeline.h>
#include <HardwareResource/SyrinxHardwareVertexBuffer.h>
#include <HardwareResource/SyrinxHardwareIndexBuffer.h>
#include <HardwareResource/SyrinxVertexInputState.h>
#include <HardwareResource/SyrinxHardwareTexture.h>
#include <Pipeline/SyrinxDisplayDevice.h>


int main(int argc, char *argv[])
{
    auto logManager = std::make_unique<Syrinx::LogManager>();

    Syrinx::DisplayDevice displayDevice;
    displayDevice.setMajorVersionNumber(4);
    displayDevice.setMinorVersionNumber(6);
    displayDevice.setDebugMessageHandler(Syrinx::DefaultDebugHandler);
    auto renderWindow = displayDevice.createWindow("Load Texture Sample", 800, 600);

    const std::string vertexProgramSource =
        "layout(location = 0) in vec3 aPos;\n"
        "layout(location = 0) out vec2 texCoord;\n"
        "out gl_PerVertex {\n"
        "    vec4 gl_Position;\n"
        "};\n"
        "void main() {\n"
        "    texCoord = aPos.xy + vec2(0.5); \n"
        "    gl_Position = vec4(aPos, 1.0);\n"
        "}\n";

    const std::string fragmentProgramSource =
        "layout(location = 0) in vec2 texCoord;\n"
        "layout(location = 0) out vec4 FragColor;\n"
        "uniform sampler2D uDiffuseTex;\n"
        "void main()\n"
        "{\n"
        "   FragColor = texture(uDiffuseTex, vec2(2.0) * texCoord);\n"
        "}\n";


    Syrinx::ProgramCompiler compiler;
    Syrinx::HardwareResourceManager hardwareResourceManager;
    auto vertexProgramBinarySource = compiler.compile("vertex", vertexProgramSource, Syrinx::ProgramStageType::VertexStage);
    auto vertexProgram = hardwareResourceManager.createProgramStage("load texture vertex program",
                                                                    std::move(vertexProgramBinarySource),
                                                                    Syrinx::ProgramStageType::VertexStage);

    auto fragmentProgramBinarySource = compiler.compile("fragment", fragmentProgramSource, Syrinx::ProgramStageType::FragmentStage);
    auto fragmentProgram = hardwareResourceManager.createProgramStage("load texture fragment program",
                                                                      std::move(fragmentProgramBinarySource),
                                                                      Syrinx::ProgramStageType::FragmentStage);

    auto programPipeline = hardwareResourceManager.createProgramPipeline("draw model program pipeline");
    programPipeline->bindProgramStage(vertexProgram);
    programPipeline->bindProgramStage(fragmentProgram);

    Syrinx::FileManager fileManager;
    fileManager.addSearchPath("../SampleMedias/");
    const std::string imageFile = "mipmap-test.png";
    auto [imageExist, filePath] = fileManager.findFile(imageFile);
    if (!imageExist) {
        SYRINX_FAULT_FMT("can not find image [{}]", imageFile);
        return 0;
    }
    auto hardwareTexture = hardwareResourceManager.createTexture(filePath, Syrinx::ImageFormat::RGB8, true);
    Syrinx::TextureViewDesc textureViewDesc;
    textureViewDesc.type = Syrinx::TextureType::TEXTURE_2D;
    textureViewDesc.levelCount = hardwareTexture->getMaxMipMapLevel();
    auto hardwareTextureView = hardwareResourceManager.createTextureView("texture view", hardwareTexture, textureViewDesc);
    SYRINX_ASSERT(hardwareTexture);
    SYRINX_ASSERT(hardwareTextureView);

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

    auto hardwareVertexBuffer = hardwareResourceManager.createVertexBuffer("quad vertex buffer", 4, 3 * sizeof(float), vertices);
    auto hardwareIndexBuffer = hardwareResourceManager.createIndexBuffer("quad index buffer", 6, Syrinx::IndexType::UINT16, indices);

    Syrinx::VertexAttributeDescription positionAttributeDescription;
    positionAttributeDescription.setLocation(0)
                              .setSemantic(Syrinx::VertexAttributeSemantic::Position)
                              .setDataType(Syrinx::VertexAttributeDataType::FLOAT3)
                              .setDataOffset(0)
                              .setBindingPoint(0);
    Syrinx::VertexAttributeLayoutDesc vertexAttributeLayoutDesc;
    vertexAttributeLayoutDesc.addVertexAttributeDesc(positionAttributeDescription);

    auto vertexInputState = hardwareResourceManager.createVertexInputState("quad vertex input state");
    vertexInputState->setVertexAttributeLayoutDesc(std::move(vertexAttributeLayoutDesc));
    vertexInputState->setVertexBuffer(0, hardwareVertexBuffer);
    vertexInputState->setIndexBuffer(hardwareIndexBuffer);
    vertexInputState->setup();

    Syrinx::SamplingSetting samplingSetting;
    samplingSetting.setBorderColor(Syrinx::Color(1.0, 0.0, 0.0, 1.0));
    samplingSetting.setWrapSMethod(Syrinx::TextureWrapMethod::CLAMP_TO_BORDER);
    samplingSetting.setWrapTMethod(Syrinx::TextureWrapMethod::CLAMP_TO_BORDER);
    samplingSetting.setWrapRMethod(Syrinx::TextureWrapMethod::CLAMP_TO_BORDER);
    samplingSetting.setMinFilterMethod(Syrinx::TextureMinFilterMethod::LINEAR_MIPMAP_LINEAR);
    samplingSetting.setMagFilterMethod(Syrinx::TextureMagFilterMethod::LINEAR);

    auto sampler = hardwareResourceManager.createSampler("border sampling", samplingSetting);
    SYRINX_ASSERT(sampler);

    auto fragmentVars = fragmentProgram->getProgramVars();
    Syrinx::SampledTexture sampledTexture(hardwareTextureView, sampler);
    fragmentVars->setTexture("uDiffuseTex", sampledTexture);
    SYRINX_ASSERT(fragmentVars);
    while (renderWindow->isOpen()) {
        float defaultValueForColorAttachment[] = {1.0, 1.0, 0.0, 1.0};
        glClearNamedFramebufferfv(0, GL_COLOR, 0, defaultValueForColorAttachment);
        float defaultDepthValue = 1.0;
        glClearNamedFramebufferfv(0, GL_DEPTH, 0, &defaultDepthValue);

        fragmentProgram->updateProgramVars(*fragmentVars);
        fragmentProgram->uploadParametersToGpu();
        fragmentProgram->bindResources();

        glBindVertexArray(vertexInputState->getHandle());
        glBindProgramPipeline(programPipeline->getHandle());
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);

        renderWindow->swapBuffer();
    }

    return 0;
}
