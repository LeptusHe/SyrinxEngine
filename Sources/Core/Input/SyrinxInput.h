#pragma once
#include <better-enums/enum.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "Common/SyrinxSingleton.h"
#include "Math/SyrinxMath.h"

namespace Syrinx {

BETTER_ENUM(KeyAction, uint8_t,
        KeyRelease = GLFW_RELEASE,
        KeyPressed = GLFW_PRESS,
        KeyRepeat = GLFW_REPEAT);


class Input : public Singleton<Input> {
public:
    using Key = int;

public:
    explicit Input(GLFWwindow* windowHandle);
    ~Input() = default;

    KeyAction getKeyAction(Key key) const;
    bool isPressed(Key key) const;
    bool isRelease(Key key) const;
    bool isRepeat(Key key) const;
    Vector2f getCursorPosition() const;

private:
    GLFWwindow *mWindowHandle;
};

} // namespace Syrinx
