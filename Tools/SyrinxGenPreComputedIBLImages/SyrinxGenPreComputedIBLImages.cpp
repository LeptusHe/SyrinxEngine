#include "SyrinxGenPreComputedIBLImages.h"
#include <Math/SyrinxMath.h>
#include <Logging/SyrinxLogManager.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>
#include <HardwareResource/SyrinxVertexInputState.h>
#include <RenderResource/SyrinxRenderTexture.h>
#include <RenderResource/SyrinxDepthTexture.h>
#include <RenderResource/SyrinxRenderTarget.h>
#include <FileSystem/SyrinxFileSystem.h>

namespace Syrinx { namespace Tool {

GenPreComputedIBLImages::GenPreComputedIBLImages(GLuint environmentMap, int environmentMapWidth)
{
    mEnvironmentMap = environmentMap;
    mEnvironmentMapWidth = environmentMapWidth;
    mWidth[MapType::IrradianceMap] = 32;
    mHeight[MapType::IrradianceMap] = 32;
    mWidth[MapType::PreFilteredMap] = 128;
    mHeight[MapType::PreFilteredMap] = 128;
    mWidth[MapType::BrdfIntegrationMap] = 512;
    mHeight[MapType::BrdfIntegrationMap] = 512;
    mFileManager = new FileManager();
    mSaveFileDirectory = "../../Medias/Textures/PreComputedIBLImages/";
    mMapTypeName[MapType::IrradianceMap] = "IrradianceMap";
    mMapTypeName[MapType::PreFilteredMap] = "PreFilteredMap";
    mMapTypeName[MapType::BrdfIntegrationMap] = "BrdfIntegrationMap";
    mCubeMapFaceName[0] = "right";
    mCubeMapFaceName[1] = "left";
    mCubeMapFaceName[2] = "top";
    mCubeMapFaceName[3] = "bottom";
    mCubeMapFaceName[4] = "back";
    mCubeMapFaceName[5] = "front";
    mImageType = ".png";
    SYRINX_ENSURE(mEnvironmentMap == environmentMap);
    SYRINX_ENSURE(mEnvironmentMapWidth > 0);
    SYRINX_ENSURE(mWidth[MapType::IrradianceMap] > 0);
    SYRINX_ENSURE(mHeight[MapType::IrradianceMap] > 0);
    SYRINX_ENSURE(mWidth[MapType::PreFilteredMap] > 0);
    SYRINX_ENSURE(mHeight[MapType::PreFilteredMap] > 0);
    SYRINX_ENSURE(mWidth[MapType::BrdfIntegrationMap] > 0);
    SYRINX_ENSURE(mHeight[MapType::BrdfIntegrationMap] > 0);
    SYRINX_ENSURE(mFileManager);
    initProgramPipeline();
    genCubeMap(MapType::IrradianceMap);
    genCubeMap(MapType::PreFilteredMap);
    genBrdfIntegrationMap();
}


GenPreComputedIBLImages::~GenPreComputedIBLImages()
{
    delete mFileManager;
    for (auto vertexProgram : mVertexProgram) {
        delete vertexProgram;
    }
    for (auto fragmentProgram : mFragmentProgram) {
        delete fragmentProgram;
    }
    for (auto pipeline : mPipeline) {
        delete pipeline;
    }
}


void GenPreComputedIBLImages::initProgramPipeline()
{
    SYRINX_EXPECT(mFileManager);
    mFileManager->addSearchPath("../../Medias");
    createProgramPipeline(MapType::IrradianceMap, "GenIrradianceMap_VS.glsl", "GenIrradianceMap_FS.glsl");
    createProgramPipeline(MapType::PreFilteredMap, "GenPreFilteredMap_VS.glsl", "GenPreFilteredMap_FS.glsl");
    createProgramPipeline(MapType::BrdfIntegrationMap, "ImagePass_VS.glsl", "GenBrdfIntegrationMap_FS.glsl");
}


const std::string GenPreComputedIBLImages::loadProgramSource(const std::string& fileName)
{
    SYRINX_EXPECT(mFileManager);
    auto fileStream = mFileManager->openFile(fileName, FileAccessMode::READ);
    return fileStream->getAsString();
}


void GenPreComputedIBLImages::createProgramPipeline(MapType mapType, const std::string& vertexShaderFileName, const std::string& fragmentShaderFileName)
{
    const std::string vertexShaderSource = loadProgramSource(vertexShaderFileName);
    const std::string fragmentShaderSource = loadProgramSource(fragmentShaderFileName);

    mVertexProgram[mapType] = new ProgramStage("vertex program");
    mVertexProgram[mapType]->setType(ProgramStageType::VertexStage);
    mVertexProgram[mapType]->setSource(vertexShaderSource);
    mVertexProgram[mapType]->create();

    mFragmentProgram[mapType] = new ProgramStage("fragment program");
    mFragmentProgram[mapType]->setType(Syrinx::ProgramStageType::FragmentStage);
    mFragmentProgram[mapType]->setSource(fragmentShaderSource);
    mFragmentProgram[mapType]->create();

    mPipeline[mapType] = new ProgramPipeline("program pipeline");
    mPipeline[mapType]->create();
    mPipeline[mapType]->bindProgramStage(mVertexProgram[mapType]);
    mPipeline[mapType]->bindProgramStage(mFragmentProgram[mapType]);

    SYRINX_ENSURE(mVertexProgram[mapType]);
    SYRINX_ENSURE(mFragmentProgram[mapType]);
    SYRINX_ENSURE(mPipeline[mapType]);
}


void GenPreComputedIBLImages::drawCube()
{
    float cubeVertices[] = {
            //立方体前面
            0.5f, 0.5f, 0.5f,
            0.5f, -0.5f, 0.5f,
            -0.5f, -0.5f, 0.5f,
            -0.5f, 0.5f, 0.5f,
            //立方体后面
            0.5f, 0.5f, -0.5f,
            0.5f, -0.5f, -0.5f,
            -0.5f, -0.5f, -0.5f,
            -0.5f, 0.5f, -0.5f,
            //立方体上面
            0.5f, 0.5f, 0.5f,
            0.5f, 0.5f, -0.5f,
            -0.5f, 0.5f, -0.5f,
            -0.5f, 0.5f, 0.5f,
            //立方体下面
            0.5f, -0.5f, 0.5f,
            0.5f, -0.5f, -0.5f,
            -0.5f, -0.5f, -0.5f,
            -0.5f, -0.5f, 0.5f,
            //立方体右面
            0.5f, 0.5f, 0.5f,
            0.5f, 0.5f, -0.5f,
            0.5f, -0.5f, -0.5f,
            0.5f, -0.5f, 0.5f,
            //立方体左面
            -0.5f, 0.5f, 0.5f,
            -0.5f, 0.5f, -0.5f,
            -0.5f, -0.5f, -0.5f,
            -0.5f, -0.5f, 0.5f,
    };

    uint16_t cubeIndices[] = {
            //立方体前面
            0, 3, 1,
            1, 3, 2,
            //立方体后面
            4, 5, 7,
            5, 6, 7,
            //立方体上面
            8, 9, 11,
            9, 10, 11,
            //立方体下面
            12, 15, 13,
            13, 15, 14,
            //立方体右面
            16, 19, 17,
            17, 19, 18,
            //立方体左面
            20, 21, 23,
            21, 22, 23
    };

    auto cubeHardwareVertexBuffer = std::make_unique<Syrinx::HardwareBuffer>("cube vertex buffer");
    Syrinx::HardwareVertexBuffer cubeVertexBuffer(std::move(cubeHardwareVertexBuffer));
    cubeVertexBuffer.setVertexNumber(24);
    cubeVertexBuffer.setVertexSizeInBytes(3 * sizeof(float));
    cubeVertexBuffer.setData(cubeVertices);
    cubeVertexBuffer.create();

    auto cubeHardwareIndexBuffer = std::make_unique<Syrinx::HardwareBuffer>("cube index buffer");
    Syrinx::HardwareIndexBuffer cubeIndexBuffer(std::move(cubeHardwareIndexBuffer));
    cubeIndexBuffer.setIndexType(Syrinx::IndexType::UINT16);
    cubeIndexBuffer.setIndexNumber(36);
    cubeIndexBuffer.setData(cubeIndices);
    cubeIndexBuffer.create();

    Syrinx::VertexAttributeDescription cubeVertexAttributeDescription(0, Syrinx::VertexAttributeSemantic::Position, Syrinx::VertexAttributeDataType::FLOAT3);
    Syrinx::VertexDataDescription cubeDataDescription(&cubeVertexBuffer, 0, 0, 3 * sizeof(float));

    Syrinx::VertexInputState cubeInputState("cube input state");
    cubeInputState.addVertexAttributeDescription(cubeVertexAttributeDescription);
    cubeInputState.addVertexDataDescription(cubeDataDescription);
    cubeInputState.addIndexBuffer(&cubeIndexBuffer);
    cubeInputState.create();

    glBindVertexArray(cubeInputState.getHandle());
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, nullptr);
    glBindVertexArray(cubeInputState.getHandle());
}


void GenPreComputedIBLImages::drawQuad()
{
    float quadVertices[] = {
            1.0f, 1.0f, 0.0f,    1.0f, 1.0f,
            -1.0f, 1.0f, 0.0f,    0.0f, 1.0f,
            1.0f,-1.0f, 0.0f,    1.0f, 0.0f,
            -1.0f,-1.0f, 0.0f,    0.0f, 0.0f
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

    glBindVertexArray(quadVertexInputState.getHandle());
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);
    glBindVertexArray(quadVertexInputState.getHandle());
}


void GenPreComputedIBLImages::genCubeMap(MapType mapType)
{
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    glViewport(0, 0, mWidth[mapType], mHeight[mapType]);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindProgramPipeline(mPipeline[mapType]->getHandle());
    glBindTextureUnit(0, mEnvironmentMap);
    GLint environmentMapLocation = glGetUniformLocation(mFragmentProgram[mapType]->getHandle(), "uEnvironmentMap");
    glProgramUniform1i(mFragmentProgram[mapType]->getHandle(), environmentMapLocation, 0);

    glm::mat4 projectionMatrix = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 100.0f);
    GLint projectMatrixLocation = glGetUniformLocation(mVertexProgram[mapType]->getHandle(), "uProjectionMatrix");
    glProgramUniformMatrix4fv(mVertexProgram[mapType]->getHandle(), projectMatrixLocation, 1, GL_FALSE, glm::value_ptr(projectionMatrix));

    switch (mapType) {
        case MapType::IrradianceMap: genIrradianceMap(); break;
        case MapType::PreFilteredMap: genPreFilteredMap(); break;
        case MapType::BrdfIntegrationMap: SYRINX_ASSERT(false && "BrdfIntegrationMap is not cubemap"); break;
        default: SYRINX_ASSERT(false && "undefined map type");
    }
}


void GenPreComputedIBLImages::genIrradianceMap()
{
    glm::mat4 viewMatrixs[6] = {
            glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
            glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
            glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
            glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
            glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
            glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
    };

    GLint viewMatrixLocation = glGetUniformLocation(mVertexProgram[MapType::IrradianceMap]->getHandle(), "uViewMatrix");
    for (unsigned int i = 0; i < 6; i++) {
        glProgramUniformMatrix4fv(mVertexProgram[MapType::IrradianceMap]->getHandle(), viewMatrixLocation, 1, GL_FALSE, glm::value_ptr(viewMatrixs[i]));
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        drawCube();
        std::string directory = mSaveFileDirectory + mMapTypeName[MapType::IrradianceMap] + "/";
        std::string fileName = mMapTypeName[MapType::IrradianceMap] + "_" + mCubeMapFaceName[i] + mImageType;
        if (!FileSystem::directoryExist(mSaveFileDirectory)) {
            FileSystem::createDirectory(mSaveFileDirectory);
            FileSystem::createDirectory(directory);
        } else if (!FileSystem::directoryExist(directory)) {
            FileSystem::createDirectory(directory);
        }
        saveFrameData2Image(directory + fileName, mWidth[MapType::IrradianceMap], mHeight[MapType::IrradianceMap]);
    }
}


void GenPreComputedIBLImages::genPreFilteredMap()
{
    GLint resolutionLocation = glGetUniformLocation(mVertexProgram[MapType::PreFilteredMap]->getHandle(), "uResolution");
    glProgramUniform1f(mVertexProgram[MapType::PreFilteredMap]->getHandle(), resolutionLocation, mEnvironmentMapWidth);
    glm::mat4 viewMatrixs[6] = {
            glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
            glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
            glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
            glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
            glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
            glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
    };

    GLint viewMatrixLocation = glGetUniformLocation(mVertexProgram[MapType::PreFilteredMap]->getHandle(), "uViewMatrix");
    unsigned int maxMipLevels = 5;
    for (unsigned int mip = 0; mip < maxMipLevels; mip++){
        auto mipWidth = static_cast<unsigned int>(mWidth[MapType::PreFilteredMap] * std::pow(0.5f, mip));
        auto mipHeight = static_cast<unsigned int>(mWidth[MapType::PreFilteredMap] * std::pow(0.5f, mip));

        glViewport(0, 0, mipWidth, mipHeight);

        float roughness = (float)mip / (float)(maxMipLevels);
        GLint roughnessLocation = glGetUniformLocation(mFragmentProgram[MapType::PreFilteredMap]->getHandle(), "uRoughness");
        glProgramUniform1f(mFragmentProgram[MapType::PreFilteredMap]->getHandle(), roughnessLocation, roughness);

        for (unsigned int i = 0; i < 6; i++) {
            glProgramUniformMatrix4fv(mVertexProgram[MapType::PreFilteredMap]->getHandle(), viewMatrixLocation, 1, GL_FALSE, glm::value_ptr(viewMatrixs[i]));
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            drawCube();
            std::string directory = mSaveFileDirectory + mMapTypeName[MapType::PreFilteredMap] + "/";
            std::string fileName = mMapTypeName[MapType::PreFilteredMap] + "_" + mCubeMapFaceName[i] + "_lv" + std::to_string(mip) + mImageType;
            if (!FileSystem::directoryExist(mSaveFileDirectory)) {
                FileSystem::createDirectory(mSaveFileDirectory);
                FileSystem::createDirectory(directory);
            } else if (!FileSystem::directoryExist(directory)) {
                FileSystem::createDirectory(directory);
            }
            saveFrameData2Image(directory + fileName, mipWidth, mipHeight);
        }
    }
}


void GenPreComputedIBLImages::genBrdfIntegrationMap()
{
    glBindProgramPipeline(mPipeline[MapType::BrdfIntegrationMap]->getHandle());
    glViewport(0, 0, mWidth[MapType::BrdfIntegrationMap], mHeight[MapType::BrdfIntegrationMap]);
    drawQuad();
    std::string directory = mSaveFileDirectory  + mMapTypeName[MapType::BrdfIntegrationMap] + "/";
    std::string fileName = mMapTypeName[MapType::BrdfIntegrationMap] + mImageType;
    if (!FileSystem::directoryExist(mSaveFileDirectory)) {
        FileSystem::createDirectory(mSaveFileDirectory);
        FileSystem::createDirectory(directory);
    } else if (!FileSystem::directoryExist(directory)) {
        FileSystem::createDirectory(directory);
    }
    saveFrameData2Image(directory + fileName, mWidth[MapType::BrdfIntegrationMap], mHeight[MapType::BrdfIntegrationMap]);
}


void GenPreComputedIBLImages::saveFrameData2Image(const std::string& imagePath, int width, int height)
{
    auto *frameData = new unsigned char[3 * width * height]();
    glReadBuffer(GL_BACK);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, frameData);
    stbi_write_png(imagePath.c_str(), width, height, 3, frameData, 0);

    delete[] frameData;
}

} // namespace Tool

} // namespace SyrinxGenPreComputedIBLImages
