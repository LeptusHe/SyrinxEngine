#pragma once
#include <Component/SyrinxController.h>
#include <Common/SyrinxAssert.h>
#include <Math/SyrinxMath.h>

class CameraMotionController : public Syrinx::Controller {
public:
    void update(TimeDelta timeDelta) override
    {
        auto& transform = getTransform();
        auto& input = getInput();
        const auto cursorPosition = getCursorOffset();

        if (!input.isPressed(GLFW_KEY_LEFT_CONTROL)) {
            return;
        }

        auto orientation = transform.getRotateMatrix();
        auto invOrientation = orientation; //glm::inverse(orientation);
        Syrinx::Vector3f forwardDir = invOrientation * Syrinx::Vector4f(0.0, 0.0, -1.0, 0.0);
        Syrinx::Vector3f rightDir = invOrientation * Syrinx::Vector4f(1.0, 0.0, 0.0, 0.0);
        Syrinx::Vector3f upDir = invOrientation * Syrinx::Vector4f(0.0, 1.0, 0.0, 0.0);

        float moveDistance = 0.01f * mMoveSpeed;
        if (input.isPressed(GLFW_KEY_A)) {
            transform.translate(-moveDistance * rightDir);
        }
        if (input.isPressed(GLFW_KEY_D)) {
            transform.translate(moveDistance * rightDir);
        }
        if (input.isPressed(GLFW_KEY_W)) {
            transform.translate(moveDistance * forwardDir);
        }
        if (input.isPressed(GLFW_KEY_S)) {
            transform.translate(-moveDistance * forwardDir);
        }

        auto eulerAngle = transform.getEulerAngle();
        eulerAngle.x -= cursorPosition.y * 0.2;
        eulerAngle.y -= cursorPosition.x * 0.2;
        transform.setEulerAngle(eulerAngle);
    }


    Syrinx::Vector2f getCursorOffset()
    {
        auto& input = getInput();

        static bool firstUpdate = true;
        if (firstUpdate) {
            mLastCursorPosition = input.getCursorPosition();
            firstUpdate = false;
            return {0, 0};
        }

        Syrinx::Vector2f currentCursorPosition = input.getCursorPosition();
        Syrinx::Vector2f cursorOffset = currentCursorPosition - mLastCursorPosition;
        mLastCursorPosition = currentCursorPosition;
        return cursorOffset;
    }

private:
    Syrinx::Vector2f mLastCursorPosition;
    float mMoveSpeed = 1.0f;
};
