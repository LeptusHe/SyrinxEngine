#pragma once
#include <algorithm>
#include <Common/SyrinxAssert.h>
#include <Component/SyrinxController.h>
#include <Math/SyrinxMath.h>

namespace Syrinx {

class CameraMotionController : public Controller {
public:
    void update(TimeDelta timeDelta) override {
        auto& transform = getTransform();
        auto& input = getInput();
        const auto cursorPosition = getCursorOffset();

        if (!input.isPressed(GLFW_KEY_LEFT_CONTROL)) {
            return;
        }

        auto orientation = transform.getRotateMatrix();
        auto invOrientation = orientation;
        Vector3f forwardDir = invOrientation * Vector4f(0.0, 0.0, -1.0, 0.0);
        Vector3f rightDir = invOrientation * Vector4f(1.0, 0.0, 0.0, 0.0);
        Vector3f upDir = invOrientation * Vector4f(0.0, 1.0, 0.0, 0.0);

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
        eulerAngle.x -= cursorPosition.y * 0.2f;
        eulerAngle.y -= cursorPosition.x * 0.2f;

        eulerAngle.x = std::max(-90.0f, eulerAngle.x);
        eulerAngle.x = std::min(90.0f, eulerAngle.x);

        transform.setEulerAngle(eulerAngle);
    }


    Vector2f getCursorOffset() {
        auto& input = getInput();

        static bool firstUpdate = true;
        if (firstUpdate) {
            mLastCursorPosition = input.getCursorPosition();
            firstUpdate = false;
            return {0, 0};
        }

        Vector2f currentCursorPosition = input.getCursorPosition();
        Vector2f cursorOffset = currentCursorPosition - mLastCursorPosition;
        mLastCursorPosition = currentCursorPosition;
        return cursorOffset;
    }

private:
    Vector2f mLastCursorPosition;
    float mMoveSpeed = 1.0f;
};

} // namespace Syrinx
