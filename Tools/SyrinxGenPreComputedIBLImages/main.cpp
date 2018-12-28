#include <fstream>
#include <sstream>
#include <Image/SyrinxImage.h>
#include <Logging/SyrinxLogManager.h>
#include <RenderPipeline/SyrinxDisplayDevice.h>
#include <RenderPipeline/SyrinxEngineSetting.h>
#include <RenderSystem/SyrinxCamera.h>
#include <HardwareResource/SyrinxProgramStage.h>
#include <HardwareResource/SyrinxProgramPipeline.h>
#include <HardwareResource/SyrinxVertexInputState.h>
#include <RenderResource/SyrinxRenderTexture.h>
#include <RenderResource/SyrinxDepthTexture.h>
#include <RenderResource/SyrinxRenderTarget.h>
#include <ResourceManager/SyrinxFileManager.h>
#include "SyrinxGenPreComputedIBLImages.h"


int main(int argc, char *argv[])
{
    Syrinx::LogManager *logManager = new Syrinx::LogManager();
    auto fileManager = new Syrinx::FileManager();

    Syrinx::DisplayDevice displayDevice;
    displayDevice.setMajorVersionNumber(4);
    displayDevice.setMinorVersionNumber(5);
    displayDevice.setDebugMessageHandler(Syrinx::DefaultDebugHandler);
    displayDevice.createWindow("IBL Cubemap Filter", 512, 512);

    Syrinx::Image images[6] = {
            Syrinx::Image("../../Medias/Textures/IrradianceMap/Indoor/right.png", Syrinx::ImageFormat::RGBAF),
            Syrinx::Image("../../Medias/Textures/IrradianceMap/Indoor/left.png", Syrinx::ImageFormat::RGBAF),
            Syrinx::Image("../../Medias/Textures/IrradianceMap/Indoor/top.png", Syrinx::ImageFormat::RGBAF),
            Syrinx::Image("../../Medias/Textures/IrradianceMap/Indoor/bottom.png", Syrinx::ImageFormat::RGBAF),
            Syrinx::Image("../../Medias/Textures/IrradianceMap/Indoor/back.png", Syrinx::ImageFormat::RGBAF),
            Syrinx::Image("../../Medias/Textures/IrradianceMap/Indoor/front.png", Syrinx::ImageFormat::RGBAF)
    };

    GLuint environmentMap;
    glGenTextures(1, &environmentMap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, environmentMap);
    for (unsigned int i = 0; i < 6; i++) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA32F, images[i].getWidth(), images[i].getHeight(), 0, GL_RGBA, GL_FLOAT, images[i].getData());
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    Syrinx::Tool::GenPreComputedIBLImages genPreComputedIBLImages(environmentMap, images[0].getWidth());

    return 0;
}