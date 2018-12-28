#include "Component/SyrinxCamera.h"
#include <Common/SyrinxAssert.h>

namespace Syrinx {

Camera::Camera(const std::string& name)
    : mName(name)
    , mPosition{0.0, 0.0, 0.0}
    , mFrontDirection{0.0, 0.0, -1.0}
    , mRightDirection{1.0, 0.0, 0.0}
    , mUpDirection{0.0, 1.0, 0.0}
    , mMoveSpeed(1.0)
{
    SYRINX_ENSURE(!mName.empty());
    updateViewMatrix();
}


void Camera::setPosition(const Point3f& position)
{
    mPosition = position;
}


void Camera::lookAt(const Point3f& position)
{
    Vector3f lookDir = Normalize(position - mPosition);
    setFrontDirection(lookDir);
}


void Camera::setFrontDirection(const Vector3f& frontDirection)
{
    mFrontDirection = frontDirection;
    updateViewMatrix();
}


void Camera::setViewportRect(const ViewportRect& viewportRect)
{
    mFrustum.setViewportRect(viewportRect);
}


void Camera::setMoveSpeed(float moveSpeed)
{
    mMoveSpeed = moveSpeed;
}


const Point3f& Camera::getPosition() const
{
    return mPosition;
}


float Camera::getMoveSpeed() const
{
    return mMoveSpeed;
}


Matrix4x4 Camera::getViewMatrix() const
{
    return glm::lookAt(mPosition, mPosition + mFrontDirection, mUpDirection);
}


Matrix4x4 Camera::getProjectionMatrix() const
{
    return mFrustum.getPerspectiveMatrix();
}


void Camera::updateViewMatrix()
{
    const Vector3f worldUp = glm::normalize(Vector3f{0.0, 1.0, 0.0});
    mFrontDirection = glm::normalize(mFrontDirection);
    mRightDirection = glm::normalize(glm::cross(mFrontDirection, worldUp));
    mUpDirection = glm::normalize(glm::cross(mRightDirection, mFrontDirection));
}

} // namespace Syrinx