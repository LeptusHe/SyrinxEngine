#include "SyrinxRenderWindow.h"
#include <Logging/SyrinxLogManager.h>

namespace Syrinx {

RenderWindow::RenderWindow()
    : mTitle("Syrinx Engine - Application")
    , mWidth(800)
    , mHeight(800)
    , mWindowHandle(nullptr)
{
    SYRINX_ENSURE(!mTitle.empty() && mWidth > 0 && mHeight > 0);
    SYRINX_ENSURE(!mWindowHandle);
}


RenderWindow::~RenderWindow()
{
    if (mWindowHandle) {
        glfwDestroyWindow(mWindowHandle);
    }
}


void RenderWindow::setTitle(const std::string& title)
{
    mTitle = title;
    SYRINX_ENSURE(!mTitle.empty() && mTitle == title);
}


void RenderWindow::setWidth(unsigned width)
{
    mWidth = width;
    SYRINX_ENSURE(mWidth > 0 && mWidth == width);
}


void RenderWindow::setHeight(unsigned height)
{
    mHeight = height;
    SYRINX_ENSURE(mHeight > 0 && mHeight == height);
}


void RenderWindow::setWindowHandle(GLFWwindow *windowHandle)
{
    SYRINX_EXPECT(windowHandle);
    mWindowHandle = windowHandle;
    SYRINX_ENSURE(mWindowHandle);
    SYRINX_ENSURE(mWindowHandle == windowHandle);
}


std::string RenderWindow::getTitle() const
{
    return mTitle;
}


unsigned RenderWindow::getWidth() const
{
    return mWidth;
}


unsigned RenderWindow::getHeight() const
{
    return mHeight;
}


const GLFWwindow* RenderWindow::getWindowHandle() const
{
    return mWindowHandle;
}


GLFWwindow* RenderWindow::fetchWindowHandle()
{
    return mWindowHandle;
}


bool RenderWindow::isOpen() const
{
    return !glfwWindowShouldClose(mWindowHandle);
}


void RenderWindow::dispatchEvents() const
{
    glfwPollEvents();
}


void RenderWindow::swapBuffer() const
{
    glfwSwapBuffers(mWindowHandle);
}

} // namespace Syrinx
