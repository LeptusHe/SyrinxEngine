#include <Math/SyrinxMath.h>
#include <Image/SyrinxImage.h>
#include <Image/SyrinxImageReader.h>
#include <Logging/SyrinxLogManager.h>
#include <RenderPipeline/SyrinxDisplayDevice.h>
#include <RenderPipeline/SyrinxEngineSetting.h>
#include <RenderSystem/SyrinxCamera.h>
#include <RenderResource/SyrinxCubeMap.h>
#include <HardwareResource/SyrinxProgramStage.h>
#include <HardwareResource/SyrinxProgramPipeline.h>
#include <HardwareResource/SyrinxVertexInputState.h>
#include <RenderResource/SyrinxRenderTexture.h>
#include <RenderResource/SyrinxDepthTexture.h>
#include <RenderResource/SyrinxRenderTarget.h>
#include <ResourceManager/SyrinxFileManager.h>

const int WINDOW_WIDTH = 800, WINDOW_HEIGHT = 800;

Syrinx::CCamera camera(glm::vec3(0.0f, 0.0f, 3.0f));

float lastX = WINDOW_WIDTH / 2.0f;
float lastY = WINDOW_HEIGHT / 2.0f;
bool firstMouse = true;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

int main(int argc, char *argv[]) {
    Syrinx::LogManager *logManager = new Syrinx::LogManager();
    auto fileManager = new Syrinx::FileManager();
    Syrinx::DisplayDevice displayDevice;

    displayDevice.setMajorVersionNumber(4);
    displayDevice.setMinorVersionNumber(5);
    displayDevice.setDebugMessageHandler(Syrinx::DefaultDebugHandler);
    auto renderWindow = displayDevice.createWindow("Skybox Sample", WINDOW_WIDTH, WINDOW_HEIGHT);

    glfwSetCursorPosCallback(displayDevice.fetchRenderWindow()->fetchWindowHandle(), mouse_callback);
    glfwSetFramebufferSizeCallback(displayDevice.fetchRenderWindow()->fetchWindowHandle(), framebuffer_size_callback);

    fileManager->addSearchPath("../../Medias/");
    auto loadProgramSource = [fileManager](const std::string& fileName) {
        auto fileStream = fileManager->openFile(fileName, Syrinx::FileAccessMode::READ);
        return fileStream->getAsString();
    };

    const std::string skyboxVertexShaderSource = loadProgramSource("Skybox_VS.glsl");
    const std::string skyboxFragmentShaderSource = loadProgramSource("Skybox_FS.glsl");

    auto skyboxVertexProgram = std::make_shared<Syrinx::ProgramStage>("Skybox vertex program");
    skyboxVertexProgram->setType(Syrinx::ProgramStageType::VertexStage);
    skyboxVertexProgram->setSource(skyboxVertexShaderSource);
    skyboxVertexProgram->create();

    auto skyboxFragmentProgram = std::make_shared<Syrinx::ProgramStage>("Skybox fragment program");
    skyboxFragmentProgram->setType(Syrinx::ProgramStageType::FragmentStage);
    skyboxFragmentProgram->setSource(skyboxFragmentShaderSource);
    skyboxFragmentProgram->create();

    Syrinx::ProgramPipeline skyboxProgramPipeline("Skybox program pipeline");
    skyboxProgramPipeline.create();
    skyboxProgramPipeline.bindProgramStage(skyboxVertexProgram.get());
    skyboxProgramPipeline.bindProgramStage(skyboxFragmentProgram.get());

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

    auto cubeHardwareVertexBuffer = std::make_unique<Syrinx::HardwareBuffer>("triangle vertex buffer");
    Syrinx::HardwareVertexBuffer cubeVertexBuffer(std::move(cubeHardwareVertexBuffer));
    cubeVertexBuffer.setVertexNumber(24);
    cubeVertexBuffer.setVertexSizeInBytes(3 * sizeof(float));
    cubeVertexBuffer.setData(cubeVertices);
    cubeVertexBuffer.create();

    auto cubeHardwareIndexBuffer = std::make_unique<Syrinx::HardwareBuffer>("triangle index buffer");
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


    Syrinx::HardwareTexture cubemapTexture("skybox cube map");
    std::vector<std::string> imageFileList = {
            "../../Medias/Textures/Skybox/right.jpg",
            "../../Medias/Textures/Skybox/left.jpg",
            "../../Medias/Textures/Skybox/top.jpg",
            "../../Medias/Textures/Skybox/bottom.jpg",
            "../../Medias/Textures/Skybox/back.jpg",
            "../../Medias/Textures/Skybox/front.jpg"
    };

    Syrinx::ImageReader imageReader;
    std::vector<Syrinx::Image> imageList;
    for (const auto& imageFile : imageFileList) {
        Syrinx::Image image = imageReader.read(imageFile, Syrinx::ImageFormat::RGBF);
        imageList.push_back(std::move(image));
    }

    Syrinx::CubeMap skybox(&cubemapTexture);
    skybox.addFaces(std::move(imageList));
    skybox.create();

    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    while (renderWindow->isOpen()) {
        auto currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        processInput(displayDevice.fetchRenderWindow()->fetchWindowHandle());

        float defaultValueForColorAttachment[] = {1.0, 1.0, 0.0, 1.0};
        glClearNamedFramebufferfv(0, GL_COLOR, 0, defaultValueForColorAttachment);
        float defaultDepthValue = 1.0;
        glClearNamedFramebufferfv(0, GL_DEPTH, 0, &defaultDepthValue);

        glBindProgramPipeline(skyboxProgramPipeline.getHandle());
        glBindVertexArray(cubeInputState.getHandle());

        glm::mat4 modelMatrix = glm::mat4(1.0f);
        glm::mat4 projectionMatrix = glm::perspective(glm::radians(45.0f), (GLfloat)(renderWindow->getWidth()) / (GLfloat)(renderWindow->getHeight()), 0.1f, 100.0f);
        glm::vec3 cameraPos = camera.getPosition();;
        glm::vec3 cameraFront = camera.getFront();
        glm::vec3 upVector = camera.getUp();
        glm::mat4 viewMatrix = glm::lookAt(cameraPos, cameraPos + cameraFront, upVector);

        skyboxVertexProgram->updateParameter("uModelMatrix", modelMatrix);
        skyboxVertexProgram->updateParameter("uViewMatrix", viewMatrix);
        skyboxVertexProgram->updateParameter("uProjectionMatrix", projectionMatrix);

        glBindTextureUnit(0, skybox.getHandle());
        skyboxFragmentProgram->updateParameter("uCubeMap", 0);
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, nullptr);
        glBindVertexArray(0);

        renderWindow->swapBuffer();
        glfwPollEvents();
    }

    return 0;
}


void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}


void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}