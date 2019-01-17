#pragma once
#include <vector>
#include <better-enums/enum.h>
#include "Math/SyrinxMath.h"
#include "Math/SyrinxPlane.h"
#include "Math/SyrinxAxisAlignedBox.h"

namespace Syrinx {


BETTER_ENUM(FrustumPlane, uint8_t, NearPlane, FarPlane, LeftPlane, RightPlane, TopPlane, BottomPlane);


class Frustum {
public:
    using FrustumPlaneList = std::vector<Plane>;

public:
    Frustum();
    ~Frustum() = default;
    void setFOVy(float FOVy);
    void setAspectRatio(float aspectRatio);
    void setNearClipDistance(float nearClipDistance);
    void setFarClipDistance(float farClipDistance);
    void setPosition(const Point3f& position);
    void lookAt(const Point3f& lookAt);
    void setFrontDir(const Vector3f& frontDir);
    const Matrix4x4& getViewMatrix();
    const Matrix4x4& getProjectionMatrix();
    float getFOVy() const;
    float getAspectRation() const;
    float getNearClipDistance() const;
    float getFarClipDistance() const;
    const Point3f& getPosition() const;
    const Vector3f& getFrontDir() const;
    bool inFrustum(const Point3f& point);
    bool inFrustum(const AxisAlignedBox& axisAlignedBox);

private:
    void updateFrustum();
    void updateViewMatrix();
    void updateProjectionMatrix();
    void updateFrustumPlaneList();
    void invalidateFrustum();

private:
    float mFOVy;
    float mAspectRatio;
    float mNearClipDistance;
    float mFarClipDistance;
    Point3f mPosition;
    Vector3f mFrontDir;
    Vector3f mUpDir;
    Vector3f mRightDir;
    Matrix4x4 mViewMatrix;
    Matrix4x4 mProjectionMatrix;
    FrustumPlaneList mFrustumPlaneList;
    bool mIsFrustumValid;
};

} // namespace Syrinx