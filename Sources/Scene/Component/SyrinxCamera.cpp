#include "Component/SyrinxCamera.h"
#include <Common/SyrinxAssert.h>

namespace Syrinx {

Camera::Camera(const std::string& name)
    : mName(name)
{
    SYRINX_ENSURE(!mName.empty());
}


Camera::Camera(const Camera& rhs)
    : mName(rhs.mName)
    , mFrustum(rhs.mFrustum)
    , mTransform(rhs.mTransform)
{

}


Camera& Camera::operator=(const Camera& rhs)
{
    if (this != &rhs) {
        mName = rhs.mName;
        mFrustum = rhs.mFrustum;
        mTransform = rhs.mTransform;
    }
    return *this;
}


std::string Camera::getName() const
{
    return mName;
}


void Camera::setTransform(const Transform *transform)
{
    SYRINX_EXPECT(transform);
    mTransform = transform;
}


void Camera::setViewportRect(const ViewportRect& viewportRect)
{
    mFrustum.setViewportRect(viewportRect);
}


Matrix4x4 Camera::getViewMatrix() const
{
    SYRINX_EXPECT(mTransform);
    const auto& worldMatrix = mTransform->getWorldMatrix();
    Vector3f position = worldMatrix * Vector4f(0.0, 0.0, 0.0, 1.0);
    Vector3f frontDir = worldMatrix * Vector4f(0.0, 0.0, -1.0, 0.0);
    frontDir = Normalize(frontDir);
    Vector3f upDir = Vector3f(0.0, 1.0, 0.0);

    return glm::lookAt(position, position + frontDir, upDir);
}


Matrix4x4 Camera::getProjectionMatrix() const
{
    return mFrustum.getPerspectiveMatrix();
}

} // namespace Syrinx