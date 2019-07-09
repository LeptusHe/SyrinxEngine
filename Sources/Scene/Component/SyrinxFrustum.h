#pragma once
#include <cstdio>
#include <Math/SyrinxMath.h>
#include <HardwareResource/SyrinxRenderTarget.h>

namespace Syrinx {

struct ViewportRect {
    ViewportRect(size_t x, size_t y, size_t width, size_t height);

    size_t x;
    size_t y;
    size_t width;
    size_t height;
};


class Frustum {
public:
    Frustum();
    ~Frustum() = default;

    void setFieldOfView(float fieldOfView);
    void setNearPlane(float nearPlane);
    void setFarPlane(float farPlane);
    void setViewportRect(const ViewportRect& viewportRect);
    void setRenderTarget(RenderTarget *renderTarget);
    float getFieldOfView() const;
    float getNearPlane() const;
    float getFarPlane() const;
    const ViewportRect& getViewportRect() const;
    RenderTarget* getRenderTarget() const;
    Matrix4x4 getPerspectiveMatrix() const;

private:
    float mFieldOfView;
    float mNearClipDistance;
    float mFarClipDistance;
    ViewportRect mViewportRect;
    RenderTarget *mRenderTarget;
};

} // namespace Syrinx