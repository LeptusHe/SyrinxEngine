#include "Component/SyrinxFrustum.h"
#include <Common/SyrinxAssert.h>

namespace Syrinx {

ViewportRect::ViewportRect(size_t x, size_t y, size_t width, size_t height)
    : x(x), y(y), width(width), height(height)
{
    SYRINX_ENSURE(width > 0);
    SYRINX_ENSURE(height > 0);
}




Frustum::Frustum()
    : mFieldOfView(90.0f)
    , mNearPlane(0.1f)
    , mFarPlane(100.0f)
    , mViewportRect(0, 0, 800, 800)
    , mRenderTarget(nullptr)
{
    SYRINX_ENSURE(!mRenderTarget);
}


void Frustum::setFieldOfView(float fieldOfView)
{
    mFieldOfView = fieldOfView;
    SYRINX_ENSURE(mFieldOfView <= 180.0f);
}


void Frustum::setNearPlane(float nearPlane)
{
    mNearPlane = nearPlane;
    SYRINX_ENSURE(mNearPlane > 0.0f);
    SYRINX_ENSURE(mNearPlane < mFarPlane);
}


void Frustum::setFarPlane(float farPlane)
{
    mFarPlane = farPlane;
    SYRINX_ENSURE(mFarPlane > 0.0f);
    SYRINX_ENSURE(mNearPlane < mFarPlane);
}


void Frustum::setViewportRect(const ViewportRect& viewportRect)
{
    mViewportRect = viewportRect;
}


void Frustum::setRenderTarget(RenderTarget *renderTarget)
{
    SYRINX_EXPECT(renderTarget);
    mRenderTarget= renderTarget;
    SYRINX_ENSURE(mRenderTarget == renderTarget);
}


float Frustum::getFieldOfView() const
{
    return mFieldOfView;
}


float Frustum::getNearPlane() const
{
    return mNearPlane;
}


float Frustum::getFarPlane() const
{
    return mFarPlane;
}


const ViewportRect& Frustum::getViewportRect() const
{
    return mViewportRect;
}


RenderTarget* Frustum::getRenderTarget() const
{
    return mRenderTarget;
}


Matrix4x4 Frustum::getPerspectiveMatrix() const
{
    auto fovy = static_cast<float>(glm::radians(mFieldOfView / 2.0));
    auto aspect = static_cast<float>(mViewportRect.width) / static_cast<float>(mViewportRect.height);
    return glm::perspective(fovy, aspect, mNearPlane, mFarPlane);
}

} // namespace Syrinx