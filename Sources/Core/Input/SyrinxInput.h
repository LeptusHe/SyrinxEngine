#pragma once
#include <better-enums/enum.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "Common/SyrinxSingleton.h"
#include "Math/SyrinxMath.h"

namespace Syrinx {

BETTER_ENUM(InputAction, uint8_t,
    Release = GLFW_RELEASE,
    Pressed = GLFW_PRESS,
    Repeat = GLFW_REPEAT);


BETTER_ENUM(MouseBotton, uint8_t,
    Left = GLFW_MOUSE_BUTTON_LEFT,
    Right = GLFW_MOUSE_BUTTON_RIGHT,
    Middle = GLFW_MOUSE_BUTTON_MIDDLE);




class Input : public Singleton<Input> {
public:
    using Key = int;

public:
    explicit Input(GLFWwindow* windowHandle);
    ~Input() = default;

    void pollInputEvents();
    InputAction getKeyAction(Key key) const;
    InputAction getMouseAction(MouseBotton mouseBotton) const;
    void setMousePos(float posX, float posY);
    bool isPressed(Key key) const;
    bool isRelease(Key key) const;
    bool isRepeat(Key key) const;
    bool isFocused() const;
    Vector2f getCursorPosition() const;

private:
    GLFWwindow *mWindowHandle;
};

} // namespace Syrinx
