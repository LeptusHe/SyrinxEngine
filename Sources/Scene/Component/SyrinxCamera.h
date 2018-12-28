#pragma once
#include <string>
#include <Math/SyrinxMath.h>
#include "Component/SyrinxFrustum.h"

namespace Syrinx {

class Camera {
public:
    explicit Camera(const std::string& name);
    ~Camera() = default;

    void setPosition(const Point3f& position);
    void lookAt(const Point3f& position);
    void setFrontDirection(const Vector3f& frontDirection);
    void setViewportRect(const ViewportRect& viewportRect);
    void setMoveSpeed(float moveSpeed);
    const Point3f& getPosition() const;
    float getMoveSpeed() const;
    Matrix4x4 getViewMatrix() const;
    Matrix4x4 getProjectionMatrix() const;

private:
    void updateViewMatrix();

private:
    std::string mName;
    Point3f mPosition;
    Vector3f mFrontDirection;
    Vector3f mRightDirection;
    Vector3f mUpDirection;
    float mMoveSpeed;
    Frustum mFrustum;
};

} // namespace Syrinx