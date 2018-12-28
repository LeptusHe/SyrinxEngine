#include "Input/SyrinxInput.h"
#include "Common/SyrinxAssert.h"

namespace Syrinx {

Input::Input(GLFWwindow *windowHandle) : mWindowHandle(windowHandle)
{
    SYRINX_ENSURE(mWindowHandle);
    SYRINX_ENSURE(mWindowHandle == windowHandle);
}


KeyAction Input::getKeyAction(Input::Key key) const
{
    auto action = glfwGetKey(mWindowHandle, key);
    return KeyAction::_from_index(static_cast<size_t>(action));
}


bool Input::isPressed(Input::Key key) const
{
    KeyAction action = getKeyAction(key);
    return action._value == KeyAction::KeyPressed;
}


bool Input::isRelease(Input::Key key) const
{
    KeyAction action = getKeyAction(key);
    return action._value == KeyAction::KeyRelease;
}


bool Input::isRepeat(Syrinx::Input::Key key) const
{
    KeyAction action = getKeyAction(key);
    return action._value == KeyAction::KeyRepeat;
}

} // namespace Syrinx