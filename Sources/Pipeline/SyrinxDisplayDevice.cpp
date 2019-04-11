#include "SyrinxDisplayDevice.h"
#include <iostream>
#include <Logging/SyrinxLogManager.h>

namespace Syrinx {

namespace {

void glfwErrorMessageHandler(int errorCode, const char *description)
{
    SYRINX_EXPECT(description);
    SYRINX_ERROR_FMT("[glfw error][error code: {}][description: {}]", errorCode, description);
};

} // anonymous namespace


DisplayDevice::DisplayDevice()
    : mMajorVersionNumber(4)
    , mMinorVersionNumber(5)
    , mDebugMessageHandler(nullptr)
    , mRenderWindow(nullptr)
{
    SYRINX_ENSURE(isVersionNumberValid());
    SYRINX_ENSURE(!mDebugMessageHandler);
    SYRINX_ENSURE(!mRenderWindow);
}


DisplayDevice::~DisplayDevice()
{
    delete mRenderWindow;
    glfwTerminate();
}


RenderWindow* DisplayDevice::createWindow(const std::string& title, unsigned width, unsigned height)
{
    SYRINX_EXPECT(!title.empty() && width > 0 && height > 0);

    bool succeedToInitWindow = initWindow(title, width, height);
    bool succeedToInit = init();
    SYRINX_ENSURE(mRenderWindow);

    if (succeedToInitWindow && succeedToInit) {
        return mRenderWindow;
    }
    return nullptr;
}


const RenderWindow* DisplayDevice::getRenderWindow() const
{
    return mRenderWindow;
}


RenderWindow* DisplayDevice::fetchRenderWindow()
{
    return mRenderWindow;
}


void DisplayDevice::setMajorVersionNumber(unsigned majorVersionNumber)
{
    mMajorVersionNumber = majorVersionNumber;
    SYRINX_ENSURE(mMajorVersionNumber == majorVersionNumber);
    SYRINX_ENSURE(isVersionNumberValid());
}


void DisplayDevice::setMinorVersionNumber(unsigned minorVersionNumber)
{
    mMinorVersionNumber = minorVersionNumber;
    SYRINX_ENSURE(mMinorVersionNumber == minorVersionNumber);
    SYRINX_ENSURE(isVersionNumberValid());
}


void DisplayDevice::setDebugMessageHandler(DebugMessageHandler debugMessageHandler)
{
    mDebugMessageHandler = debugMessageHandler;
    SYRINX_ENSURE(mDebugMessageHandler == debugMessageHandler);
}


unsigned DisplayDevice::getMajorVersionNumber() const
{
    return mMajorVersionNumber;
}


unsigned DisplayDevice::getMinorVersionNumber() const
{
    return mMinorVersionNumber;
}


DebugMessageHandler DisplayDevice::getDebugMessageHandler() const
{
    return mDebugMessageHandler;
}


bool DisplayDevice::initWindow(const std::string& title, unsigned width, unsigned height)
{
    SYRINX_EXPECT(!mRenderWindow && isVersionNumberValid());
    SYRINX_EXPECT(!title.empty() && width > 0 && height > 0);

    glfwSetErrorCallback(glfwErrorMessageHandler);
    if (glfwInit() == GLFW_TRUE) {
        SYRINX_INFO("succeed to create glfw");
    } else {
        SYRINX_FAULT("fail to create glfw");
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, mMajorVersionNumber);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, mMinorVersionNumber);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, mDebugMessageHandler != nullptr);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    GLFWwindow *windowHandle = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);

    if (windowHandle) {
        glfwMakeContextCurrent(windowHandle);
        mRenderWindow = new RenderWindow();
        mRenderWindow->setWindowHandle(windowHandle);
        mRenderWindow->setTitle(title);
        mRenderWindow->setWidth(width);
        mRenderWindow->setHeight(height);
        SYRINX_INFO_FMT("succeed to create window [title={}, major version={}, minor version={}]", title, mMajorVersionNumber, mMinorVersionNumber);
    } else {
        SYRINX_FAULT_FMT("fail to create window [title={}, major version={}, minor version={}]", title, mMajorVersionNumber, mMinorVersionNumber);
        glfwTerminate();
        return false;
    }
    SYRINX_ENSURE(windowHandle);
    return !(mDebugMessageHandler && !isDebugContextCreated());
}


bool DisplayDevice::init()
{
    bool isFunctionLoaded = loadFunctions();
    bool isMessageHandlerInitialized = false;
    if (!mDebugMessageHandler) {
        SYRINX_INFO("debug message undefined");
    } else {
        isMessageHandlerInitialized = initDebugMessageHandler();
    }
    return isFunctionLoaded && isMessageHandlerInitialized;
}


bool DisplayDevice::loadFunctions()
{
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        SYRINX_FAULT("fail to load functions");
        return false;
    } else {
        SYRINX_INFO("succeed to load functions");
        return true;
    }
}


bool DisplayDevice::initDebugMessageHandler()
{
    if (!glDebugMessageCallback) {
        SYRINX_ERROR("fail to set debug message handler, because glMessageCallback is not available");
        return false;
    }
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(mDebugMessageHandler, nullptr);
    GLuint unusedIds = 0;
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, &unusedIds, GL_TRUE);

    SYRINX_INFO("succeed to set debug message handler");
    return true;
}


bool DisplayDevice::isDebugContextCreated() const
{
    GLint contextFlag = 0;
    glGetIntegerv(GL_CONTEXT_FLAGS, &contextFlag);
    if (contextFlag & GL_CONTEXT_FLAG_DEBUG_BIT) {
        SYRINX_INFO("succeed to create debug context");
        return true;
    } else {
        SYRINX_ERROR("fail to create debug context");
        return false;
    }
}


bool DisplayDevice::isVersionNumberValid() const
{
    return mMajorVersionNumber == 4 && mMinorVersionNumber >= 5;
}

} // namespace Syrinx