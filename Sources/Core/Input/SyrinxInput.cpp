#include "Input/SyrinxInput.h"
#include "Common/SyrinxAssert.h"

namespace Syrinx {

Input::Input(GLFWwindow *windowHandle) : mWindowHandle(windowHandle)
{
    SYRINX_ENSURE(mWindowHandle);
    SYRINX_ENSURE(mWindowHandle == windowHandle);
}


void Input::pollInputEvents()
{
    glfwPollEvents();
}


InputAction Input::getKeyAction(Input::Key key) const
{
    auto action = glfwGetKey(mWindowHandle, key);
    return InputAction::_from_index(static_cast<size_t>(action));
}


InputAction Input::getMouseAction(MouseBotton mouseBotton) const
{
    auto action = glfwGetMouseButton(mWindowHandle, mouseBotton._value);
    return InputAction::_from_index(static_cast<size_t>(action));
}


void Input::setMousePos(float posX, float posY)
{
    glfwSetCursorPos(mWindowHandle, posX, posY);
}


bool Input::isPressed(Input::Key key) const
{
    InputAction action = getKeyAction(key);
    return action._value == InputAction::Pressed;
}


bool Input::isRelease(Input::Key key) const
{
    InputAction action = getKeyAction(key);
    return action._value == InputAction::Release;
}


bool Input::isRepeat(Input::Key key) const
{
    InputAction action = getKeyAction(key);
    return action._value == InputAction::Repeat;
}


bool Input::isFocused() const
{
    return glfwGetWindowAttrib(mWindowHandle, GLFW_FOCUSED) != 0;
}


Vector2f Input::getCursorPosition() const
{
    double xPos = 0.0;
    double yPos = 0.0;
    glfwGetCursorPos(mWindowHandle, &xPos, &yPos);
    return {xPos, yPos};
}

} // namespace Syrinx