#pragma once
#include <Component/SyrinxController.h>
#include <Common/SyrinxAssert.h>

class CubeMotionController : public Syrinx::Controller {
public:
    void update(TimeDelta timeDelta) override
    {
        auto& transform = getTransform();
        auto& input = getInput();
        if (input.isPressed(GLFW_KEY_A)) {
            transform.translate({-0.01, 0.0, 0.0});
        }
        if (input.isPressed(GLFW_KEY_D)) {
            transform.translate({0.01, 0.0, 0.0});
        }
        if (input.isPressed(GLFW_KEY_W)) {
            transform.translate({0.0, 0.01, 0.0});
        }
        if (input.isPressed(GLFW_KEY_S)) {
            transform.translate({0.0, -0.01, 0.0});
        }
        if (input.isPressed(GLFW_KEY_J)) {
            transform.translate({0.0, 0.0, -0.01});
        }
        if (input.isPressed(GLFW_KEY_K)) {
            transform.translate({0.0, 0.0, 0.01});
        }
    }
};


