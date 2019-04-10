#include <fstream>
#include <sstream>
#include <Image/SyrinxImage.h>
#include <Image/SyrinxImageReader.h>
#include <HardwareResource/SyrinxHardwareTexture.h>
#include <Logging/SyrinxLogManager.h>
#include <ResourceManager/SyrinxFileManager.h>
#include <HardwareResource/SyrinxProgramStage.h>
#include <HardwareResource/SyrinxProgramPipeline.h>
#include <HardwareResource/SyrinxHardwareVertexBuffer.h>
#include <HardwareResource/SyrinxHardwareIndexBuffer.h>
#include <HardwareResource/SyrinxVertexInputState.h>
#include <Pipeline/SyrinxDisplayDevice.h>
#include <RenderResource/SyrinxRenderTexture.h>
#include <RenderResource/SyrinxDepthTexture.h>
#include <RenderResource/SyrinxRenderTarget.h>

const int WIDTH = 1024, HEIGHT = 768;

const std::string loadShaderSource(const GLchar* shaderFilePath);


int main(int argc, char *argv[])
{
    Syrinx::LogManager *logManager = new Syrinx::LogManager();
    auto fileManager = new Syrinx::FileManager();
    Syrinx::DisplayDevice displayDevice;

    displayDevice.setMajorVersionNumber(4);
    displayDevice.setMinorVersionNumber(5);
    displayDevice.setDebugMessageHandler(Syrinx::DefaultDebugHandler);

    const int WINDOW_WIDTH = 1024;
    const int WINDOW_HEIGHT = 768;
    displayDevice.createWindow("ToneMapping", WIDTH, HEIGHT);
    auto renderWindow = displayDevice.getRenderWindow();

    fileManager->addSearchPath("../../Medias/");
    auto loadProgramSource = [fileManager](const std::string& fileName) {
        auto fileStream = fileManager->openFile(fileName, Syrinx::FileAccessMode::READ);
        return fileStream->getAsString();
    };

    //--------------------------------downsampling shader-------------------------------------
    const std::string downsampling4XVertexShaderSource = loadProgramSource("Downsampling4X_VS.glsl");
    const std::string downsampling4XFragmentShaderSource = loadProgramSource("Downsampling4X_FS.glsl");

    auto downsampling4XVertexProgram = std::make_shared<Syrinx::ProgramStage>("downsampling vertex program");
    downsampling4XVertexProgram->setType(Syrinx::ProgramStageType::VertexStage);
    downsampling4XVertexProgram->setSource(downsampling4XVertexShaderSource);
    downsampling4XVertexProgram->create();

    auto downsampling4XFragmentProgram = std::make_shared<Syrinx::ProgramStage>("downsampling fragment program");
    downsampling4XFragmentProgram->setType(Syrinx::ProgramStageType::FragmentStage);
    downsampling4XFragmentProgram->setSource(downsampling4XFragmentShaderSource);
    downsampling4XFragmentProgram->create();

    Syrinx::ProgramPipeline downsampling4XProgramPipeline("downsampling program pipeline");
    downsampling4XProgramPipeline.create();
    downsampling4XProgramPipeline.bindProgramStage(downsampling4XVertexProgram.get());
    downsampling4XProgramPipeline.bindProgramStage(downsampling4XFragmentProgram.get());

    //--------------------------------calculate luminance shader-------------------------------------
    const std::string calcLuminanceComputeShaderSource = loadProgramSource("CalculateLuminance_CS.glsl");

    auto calcLuminanceComputeProgram = std::make_shared<Syrinx::ProgramStage>("calculate luminance compute program");
    calcLuminanceComputeProgram->setType(Syrinx::ProgramStageType::ComputeStage);
    calcLuminanceComputeProgram->setSource(calcLuminanceComputeShaderSource);
    calcLuminanceComputeProgram->create();

    Syrinx::ProgramPipeline calcLuminanceProgramPipeline("calculate luminance program pipeline");
    calcLuminanceProgramPipeline.create();
    calcLuminanceProgramPipeline.bindProgramStage(calcLuminanceComputeProgram.get());

    //--------------------------------calculate adapted luminance shader-------------------------------------
    const std::string calcAdaptedLuminanceComputeShaderSource = loadProgramSource("CalculateAdaptedLuminance_CS.glsl");

    auto calcAdaptedLuminanceComputeProgram = std::make_shared<Syrinx::ProgramStage>("calculate adapted luminance compute program");
    calcAdaptedLuminanceComputeProgram->setType(Syrinx::ProgramStageType::ComputeStage);
    calcAdaptedLuminanceComputeProgram->setSource(calcAdaptedLuminanceComputeShaderSource);
    calcAdaptedLuminanceComputeProgram->create();

    Syrinx::ProgramPipeline calcAdaptedLuminanceProgramPipeline("calculate adapted luminance program pipeline");
    calcAdaptedLuminanceProgramPipeline.create();
    calcAdaptedLuminanceProgramPipeline.bindProgramStage(calcAdaptedLuminanceComputeProgram.get());

    //--------------------------------copy luminance shader-------------------------------------
    const std::string copyLuminanceComputeShaderSource = loadProgramSource("CopyLuminance_CS.glsl");

    auto copyLuminanceComputeProgram = std::make_shared<Syrinx::ProgramStage>("copy luminance fragment program");
    copyLuminanceComputeProgram->setType(Syrinx::ProgramStageType::ComputeStage);
    copyLuminanceComputeProgram->setSource(copyLuminanceComputeShaderSource);
    copyLuminanceComputeProgram->create();

    Syrinx::ProgramPipeline copyLuminanceProgramPipeline("copy luminance program pipeline");
    copyLuminanceProgramPipeline.create();
    copyLuminanceProgramPipeline.bindProgramStage(copyLuminanceComputeProgram.get());

    //--------------------------------tone mapping shader-------------------------------------
    const std::string imagePassVertexShaderSource = loadProgramSource("ImagePass_VS.glsl");
    const std::string toneMappingFragmentShaderSource = loadProgramSource("ToneMapping_FS.glsl");

    auto toneMappingVertexProgram = std::make_shared<Syrinx::ProgramStage>("tone mapping vertex program");
    toneMappingVertexProgram->setType(Syrinx::ProgramStageType::VertexStage);
    toneMappingVertexProgram->setSource(imagePassVertexShaderSource);
    toneMappingVertexProgram->create();

    auto toneMappingFragmentProgram = std::make_shared<Syrinx::ProgramStage>("tone mapping fragment program");
    toneMappingFragmentProgram->setType(Syrinx::ProgramStageType::FragmentStage);
    toneMappingFragmentProgram->setSource(toneMappingFragmentShaderSource);
    toneMappingFragmentProgram->create();

    Syrinx::ProgramPipeline toneMappingProgramPipeline("tone mapping program pipeline");
    toneMappingProgramPipeline.create();
    toneMappingProgramPipeline.bindProgramStage(toneMappingVertexProgram.get());
    toneMappingProgramPipeline.bindProgramStage(toneMappingFragmentProgram.get());

    //--------------------------------texture copy pass-------------------------------------
    const std::string textureCopyFragmentShaderSource = loadProgramSource("TextureCopy_FS.glsl");

    auto textureCopyVertexProgram = std::make_shared<Syrinx::ProgramStage>("texture copy fragment program");
    textureCopyVertexProgram->setType(Syrinx::ProgramStageType::VertexStage);
    textureCopyVertexProgram->setSource(imagePassVertexShaderSource);
    textureCopyVertexProgram->create();

    auto textureCopyFragmentProgram = std::make_shared<Syrinx::ProgramStage>("texture copy fragment program");
    textureCopyFragmentProgram->setType(Syrinx::ProgramStageType::FragmentStage);
    textureCopyFragmentProgram->setSource(textureCopyFragmentShaderSource);
    textureCopyFragmentProgram->create();

    Syrinx::ProgramPipeline textureCopyProgramPipeline("texture copy program pipeline");
    textureCopyProgramPipeline.create();
    textureCopyProgramPipeline.bindProgramStage(textureCopyVertexProgram.get());
    textureCopyProgramPipeline.bindProgramStage(textureCopyFragmentProgram.get());

    //--------------------------------texture display shader-------------------------------------
    const std::string textureDisplayFragmentShaderSource = loadProgramSource("TextureDisplayWithGammaCorrection_FS.glsl");

    auto textureDisplayVertexProgram = std::make_shared<Syrinx::ProgramStage>("texture display fragment program");
    textureDisplayVertexProgram->setType(Syrinx::ProgramStageType::VertexStage);
    textureDisplayVertexProgram->setSource(imagePassVertexShaderSource);
    textureDisplayVertexProgram->create();

    auto textureDisplayFragmentProgram = std::make_shared<Syrinx::ProgramStage>("texture display fragment program");
    textureDisplayFragmentProgram->setType(Syrinx::ProgramStageType::FragmentStage);
    textureDisplayFragmentProgram->setSource(textureDisplayFragmentShaderSource);
    textureDisplayFragmentProgram->create();

    Syrinx::ProgramPipeline textureDisplayProgramPipeline("texture display program pipeline");
    textureDisplayProgramPipeline.create();
    textureDisplayProgramPipeline.bindProgramStage(textureDisplayVertexProgram.get());
    textureDisplayProgramPipeline.bindProgramStage(textureDisplayFragmentProgram.get());

	//--------------------------------quad VAO -------------------------------------
    float quadVertices[] = {
            1.0f, 1.0f, 0.0f,       1.0f, 1.0f,
            -1.0f, 1.0f, 0.0f,       0.0f, 1.0f,
            1.0f,-1.0f, 0.0f,       1.0f, 0.0f,
            -1.0f,-1.0f, 0.0f,       0.0f, 0.0f
    };

    uint16_t quadIndices[] = {
            0,1, 2,
            2, 1, 3
    };

    auto vertexBuffer = std::make_unique<Syrinx::HardwareBuffer>("quad vertex buffer");
    Syrinx::HardwareVertexBuffer hardwareVertexBuffer(std::move(vertexBuffer));
    hardwareVertexBuffer.setVertexNumber(4);
    hardwareVertexBuffer.setVertexSizeInBytes(5 * sizeof(float));
    hardwareVertexBuffer.setData(quadVertices);
    hardwareVertexBuffer.create();

    auto indexBuffer = std::make_unique<Syrinx::HardwareBuffer>("quad index buffer");
    Syrinx::HardwareIndexBuffer hardwareIndexBuffer(std::move(indexBuffer));
    hardwareIndexBuffer.setIndexType(Syrinx::IndexType::UINT16);
    hardwareIndexBuffer.setIndexNumber(6);
    hardwareIndexBuffer.setData(quadIndices);
    hardwareIndexBuffer.create();

    Syrinx::VertexAttributeDescription vertexAttributePositionDescription(0, Syrinx::VertexAttributeSemantic::Position, Syrinx::VertexAttributeDataType::FLOAT3);
    Syrinx::VertexAttributeDescription vertexAttributeTexCoordDescription(1, Syrinx::VertexAttributeSemantic::TexCoord, Syrinx::VertexAttributeDataType::FLOAT2);
    Syrinx::VertexDataDescription vertexDataPositionDescription(&hardwareVertexBuffer, 0, 0, 5 * sizeof(float));
    Syrinx::VertexDataDescription vertexDataTexCoordDescription(&hardwareVertexBuffer, 1, 3 * sizeof(float), 5 * sizeof(float));

    Syrinx::VertexInputState vertexInputState("quad vertex input state");
    vertexInputState.addVertexAttributeDescription(vertexAttributePositionDescription);
    vertexInputState.addVertexAttributeDescription(vertexAttributeTexCoordDescription);
    vertexInputState.addVertexDataDescription(vertexDataPositionDescription);
    vertexInputState.addVertexDataDescription(vertexDataTexCoordDescription);
    vertexInputState.addIndexBuffer(&hardwareIndexBuffer);
    vertexInputState.create();

	//--------------------------------Texture -------------------------------------
	Syrinx::ImageReader imageReader;
	Syrinx::Image image = imageReader.read("../../Medias/withoutToneMapping.png", Syrinx::ImageFormat::RGBA8);

    Syrinx::HardwareTexture skyTexture("without tone mapping texture");
    skyTexture.setType(Syrinx::TextureType::TEXTURE_2D);
    skyTexture.setPixelFormat(Syrinx::PixelFormat::RGBA8);
    skyTexture.setWidth(image.getWidth());
    skyTexture.setHeight(image.getHeight());
    skyTexture.create();
    skyTexture.write(image.getData(), image.getWidth(), image.getHeight());

    Syrinx::HardwareTexture avgLuminanceTexture("average luminance texture");
    avgLuminanceTexture.setType(Syrinx::TextureType::TEXTURE_2D);
    avgLuminanceTexture.setPixelFormat(Syrinx::PixelFormat::RGBAF);
    avgLuminanceTexture.setWidth(1);
    avgLuminanceTexture.setHeight(1);
    avgLuminanceTexture.create();

    Syrinx::HardwareTexture currentLuminanceTexture("current luminance texture");
    currentLuminanceTexture.setType(Syrinx::TextureType::TEXTURE_2D);
    currentLuminanceTexture.setPixelFormat(Syrinx::PixelFormat::RGBAF);
    currentLuminanceTexture.setWidth(1);
    currentLuminanceTexture.setHeight(1);
    currentLuminanceTexture.create();

    Syrinx::HardwareTexture luminanceTextureA("luminance texture A");
    luminanceTextureA.setType(Syrinx::TextureType::TEXTURE_2D);
    luminanceTextureA.setPixelFormat(Syrinx::PixelFormat::RGBAF);
    luminanceTextureA.setWidth(1);
    luminanceTextureA.setHeight(1);
    luminanceTextureA.create();

    Syrinx::HardwareTexture luminanceTextureB("luminance texture B");
    luminanceTextureB.setType(Syrinx::TextureType::TEXTURE_2D);
    luminanceTextureB.setPixelFormat(Syrinx::PixelFormat::RGBAF);
    luminanceTextureB.setWidth(1);
    luminanceTextureB.setHeight(1);
    luminanceTextureB.create();

    //--------------------------------downsampling FBO -------------------------------------
    Syrinx::HardwareTexture downsampling4XTexture("downsampling texture");
    downsampling4XTexture.setType(Syrinx::TextureType::TEXTURE_2D);
    downsampling4XTexture.setPixelFormat(Syrinx::PixelFormat::RGBAF);
    downsampling4XTexture.setWidth(256);
    downsampling4XTexture.setHeight(256);
    downsampling4XTexture.create();

    Syrinx::HardwareTexture downsampling4XDepthTexture("downsampling depth texture");
    downsampling4XDepthTexture.setType(Syrinx::TextureType::TEXTURE_2D);
    downsampling4XDepthTexture.setPixelFormat(Syrinx::PixelFormat::DEPTH24);
    downsampling4XDepthTexture.setWidth(256);
    downsampling4XDepthTexture.setHeight(256);
    downsampling4XDepthTexture.create();

    Syrinx::RenderTexture downsampling4XRenderTexture("downsampling4X render texture", &downsampling4XTexture);
    Syrinx::RenderTexture downsampling4XDepthRenderTexture("downsampling4X depth render texture", &downsampling4XDepthTexture);
    Syrinx::DepthTexture downsampling4XDepthAttachment("downsampling4X depth attachment", &downsampling4XDepthRenderTexture);

    Syrinx::RenderTarget downsampling4XRenderTarget1("downsampling4X render target 1");
    downsampling4XRenderTarget1.addRenderTexture(0, &downsampling4XRenderTexture);
    downsampling4XRenderTarget1.addDepthTexture(&downsampling4XDepthAttachment);
    downsampling4XRenderTarget1.create();

    //--------------------------------downsampling FBO 2-------------------------------------
    Syrinx::HardwareTexture exposureTextureA("exposure texture A");
    exposureTextureA.setType(Syrinx::TextureType::TEXTURE_2D);
    exposureTextureA.setPixelFormat(Syrinx::PixelFormat::RGBAF);
    exposureTextureA.setWidth(64);
    exposureTextureA.setHeight(64);
    exposureTextureA.create();

    Syrinx::HardwareTexture exposureDepthTextureA("exposure depth texture A");
    exposureDepthTextureA.setType(Syrinx::TextureType::TEXTURE_2D);
    exposureDepthTextureA.setPixelFormat(Syrinx::PixelFormat::DEPTH24);
    exposureDepthTextureA.setWidth(64);
    exposureDepthTextureA.setHeight(64);
    exposureDepthTextureA.create();

    Syrinx::RenderTexture exposureRenderTextureA("exposure render texture A", &exposureTextureA);
    Syrinx::RenderTexture exposureDepthRenderTextureA("exposure depth render texture A", &exposureDepthTextureA);
    Syrinx::DepthTexture exposureDepthAttachmentA("exposure depth attachment A", &exposureDepthRenderTextureA);

    Syrinx::RenderTarget downsampling4XRenderTarget2("downsampling4X render target 2");
    downsampling4XRenderTarget2.addRenderTexture(0, &exposureRenderTextureA);
    downsampling4XRenderTarget2.addDepthTexture(&exposureDepthAttachmentA);
    downsampling4XRenderTarget2.create();

    //--------------------------------downsampling FBO 3-------------------------------------
    Syrinx::HardwareTexture exposureTextureB("exposure texture B");
    exposureTextureB.setType(Syrinx::TextureType::TEXTURE_2D);
    exposureTextureB.setPixelFormat(Syrinx::PixelFormat::RGBAF);
    exposureTextureB.setWidth(16);
    exposureTextureB.setHeight(16);
    exposureTextureB.create();

    Syrinx::HardwareTexture exposureDepthTextureB("exposure depth texture B");
    exposureDepthTextureB.setType(Syrinx::TextureType::TEXTURE_2D);
    exposureDepthTextureB.setPixelFormat(Syrinx::PixelFormat::DEPTH24);
    exposureDepthTextureB.setWidth(16);
    exposureDepthTextureB.setHeight(16);
    exposureDepthTextureB.create();

    Syrinx::RenderTexture exposureRenderTextureB("exposure render texture B", &exposureTextureB);
    Syrinx::RenderTexture exposureDepthRenderTextureB("exposure depth render texture B", &exposureDepthTextureB);
    Syrinx::DepthTexture exposureDepthAttachmentB("exposure depth attachment B", &exposureDepthRenderTextureB);

    Syrinx::RenderTarget downsampling4XRenderTarget3("downsampling4X render target 3");
    downsampling4XRenderTarget3.addRenderTexture(0, &exposureRenderTextureB);
    downsampling4XRenderTarget3.addDepthTexture(&exposureDepthAttachmentB);
    downsampling4XRenderTarget3.create();

    //--------------------------------toneMapping FBO-------------------------------------
    Syrinx::HardwareTexture sceneColorMidTexture("scene color mid texture");
    sceneColorMidTexture.setType(Syrinx::TextureType::TEXTURE_2D);
    sceneColorMidTexture.setPixelFormat(Syrinx::PixelFormat::RGBAF);
    sceneColorMidTexture.setWidth(renderWindow->getWidth());
    sceneColorMidTexture.setHeight(renderWindow->getHeight());
    sceneColorMidTexture.create();

    Syrinx::HardwareTexture sceneDepthTexture("scene depth texture");
    sceneDepthTexture.setType(Syrinx::TextureType::TEXTURE_2D);
    sceneDepthTexture.setPixelFormat(Syrinx::PixelFormat::DEPTH24);
    sceneDepthTexture.setWidth(renderWindow->getWidth());
    sceneDepthTexture.setHeight(renderWindow->getHeight());
    sceneDepthTexture.create();

    Syrinx::RenderTexture sceneColorMidRenderTexture("scene color mid render texture", &sceneColorMidTexture);
    Syrinx::RenderTexture sceneDepthRenderTexture("scene depth render texture", &sceneDepthTexture);
    Syrinx::DepthTexture sceneDepthAttachment("scene depth attachment", &sceneDepthRenderTexture);

    Syrinx::RenderTarget toneMappingRenderTarget("toneMapping render target");
    toneMappingRenderTarget.addRenderTexture(0, &sceneColorMidRenderTexture);
    toneMappingRenderTarget.addDepthTexture(&sceneDepthAttachment);
    toneMappingRenderTarget.create();

    //--------------------------------texture copy FBO-------------------------------------
    Syrinx::HardwareTexture sceneColorTexture("scene color texture");
    sceneColorTexture.setType(Syrinx::TextureType::TEXTURE_2D);
    sceneColorTexture.setPixelFormat(Syrinx::PixelFormat::RGBAF);
    sceneColorTexture.setWidth(WIDTH);
    sceneColorTexture.setHeight(HEIGHT);
    sceneColorTexture.create();

    Syrinx::RenderTexture sceneColorRenderTexture("scene color render texture", &sceneColorTexture);

    Syrinx::RenderTarget textureCopyRenderTarget("texture copy render target");
    textureCopyRenderTarget.addRenderTexture(0, &sceneColorRenderTexture);
    textureCopyRenderTarget.addDepthTexture(&sceneDepthAttachment);
    textureCopyRenderTarget.create();

    bool swapLuminanceTexture = true;
    float defaultDepthValue = 1.0;
    float defaultValueForColorAttachment[] = {0.0, 0.0, 1.0, 1.0};
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    while (renderWindow->isOpen()) {
        swapLuminanceTexture = !swapLuminanceTexture;
        //--------------------------------Downsampling pass-------------------------------------
        glBindFramebuffer(GL_FRAMEBUFFER, downsampling4XRenderTarget1.getHandle());
        glBindVertexArray(vertexInputState.getHandle());
        glBindProgramPipeline(downsampling4XProgramPipeline.getHandle());
        glBindTextureUnit(0, skyTexture.getHandle());
        downsampling4XFragmentProgram->updateParameter("uTexSampler", 0);
        glViewport(0, 0, 256, 256);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);

        glBindFramebuffer(GL_FRAMEBUFFER, downsampling4XRenderTarget2.getHandle());
        glBindTextureUnit(0, downsampling4XTexture.getHandle());
        glViewport(0, 0, 64, 64);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);

        glBindFramebuffer(GL_FRAMEBUFFER, downsampling4XRenderTarget3.getHandle());
        glBindTextureUnit(0, exposureTextureA.getHandle());
        glViewport(0, 0, 16, 16);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);
        glBindVertexArray(0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        //--------------------------------Calculate Luminance pass-------------------------------------
        glBindProgramPipeline(calcLuminanceProgramPipeline.getHandle());
        glBindImageTexture(0, exposureTextureB.getHandle(), 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
        glBindImageTexture(1, currentLuminanceTexture.getHandle(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
        calcLuminanceComputeProgram->updateParameter("uInputImage", 0);
        calcLuminanceComputeProgram->updateParameter("uOutputImage", 1);
        glDispatchCompute(1, 1, 1);

        glBindProgramPipeline(calcAdaptedLuminanceProgramPipeline.getHandle());
        glBindImageTexture(0, currentLuminanceTexture.getHandle(), 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
        if(!swapLuminanceTexture) {
            glBindImageTexture(1, luminanceTextureA.getHandle(), 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
            glBindImageTexture(2, luminanceTextureB.getHandle(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
        }else{
            glBindImageTexture(1, luminanceTextureB.getHandle(), 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
            glBindImageTexture(2, luminanceTextureA.getHandle(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
        }
        calcAdaptedLuminanceComputeProgram->updateParameter("uCurrentImage", 0);
        calcAdaptedLuminanceComputeProgram->updateParameter("uImage0", 1);
        calcAdaptedLuminanceComputeProgram->updateParameter("uImage1", 2);
        glDispatchCompute(1, 1, 1);

        glBindProgramPipeline(copyLuminanceProgramPipeline.getHandle());
        if(!swapLuminanceTexture){
            glBindImageTexture(0, luminanceTextureB.getHandle(), 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
        }else{
            glBindImageTexture(0, luminanceTextureA.getHandle(), 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
        }
        glBindImageTexture(1, avgLuminanceTexture.getHandle(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
        copyLuminanceComputeProgram->updateParameter("uInputImage", 0);
        copyLuminanceComputeProgram->updateParameter("uOutputImage", 1);
        glDispatchCompute(1, 1, 1);

        //--------------------------------Tone Mapping pass-------------------------------------
        glBindFramebuffer(GL_FRAMEBUFFER, toneMappingRenderTarget.getHandle());
        glBindProgramPipeline(toneMappingProgramPipeline.getHandle());
        glBindVertexArray(vertexInputState.getHandle());
        glBindTextureUnit(0, skyTexture.getHandle());
        glBindTextureUnit(1, avgLuminanceTexture.getHandle());
        toneMappingFragmentProgram->updateParameter("uTexSampler", 0);
        toneMappingFragmentProgram->updateParameter("uAvgLuminanceSampler", 1);
        toneMappingFragmentProgram->updateParameter("uExposure", 1.0f);
        glViewport(0, 0, WIDTH, HEIGHT);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);

        //--------------------------------Texture Copy pass-------------------------------------
        glBindFramebuffer(GL_FRAMEBUFFER, textureCopyRenderTarget.getHandle());
        glBindProgramPipeline(textureCopyProgramPipeline.getHandle());
        glBindTextureUnit(0, sceneColorMidTexture.getHandle());
        textureCopyFragmentProgram->updateParameter("uTexSampler", 0);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);

        //--------------------------------Display Final Result pass-------------------------------------
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClearNamedFramebufferfv(0, GL_COLOR, 0, defaultValueForColorAttachment);
        glClearNamedFramebufferfv(0, GL_DEPTH, 0, &defaultDepthValue);
        glBindProgramPipeline(textureDisplayProgramPipeline.getHandle());
        glBindVertexArray(vertexInputState.getHandle());
        glBindTextureUnit(0, sceneColorTexture.getHandle());
        textureDisplayFragmentProgram->updateParameter("uTexSampler", 0);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);
        glBindVertexArray(0);
        renderWindow->swapBuffer();
    }

    return 0;
}


const std::string loadShaderSource(const GLchar* shaderFilePath)
{
    std::string ShaderSource;
    std::ifstream ShaderSourceStream;
    // 保证ifstream对象可以抛出异常
    ShaderSourceStream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try
    {
        ShaderSourceStream.open(shaderFilePath);
        std::stringstream ShaderStream;
        ShaderStream << ShaderSourceStream.rdbuf();
        ShaderSourceStream.close();
        ShaderSource = ShaderStream.str();
        return ShaderSource;
    }
    catch (std::ifstream::failure e)
    {
        SYRINX_ERROR_FMT("fail to load shader text: [{}]", shaderFilePath);
        return std::string();
    }
}