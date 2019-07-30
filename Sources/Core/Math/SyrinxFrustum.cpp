#include "Math/SyrinxFrustum.h"
#include "Common/SyrinxAssert.h"

namespace Syrinx {

Frustum::Frustum()
    : mFOVy(45.0f)
    , mAspectRatio(1.0f)
    , mNearClipDistance(0.1f)
    , mFarClipDistance(100.0f)
    , mPosition(0.0f, 0.0f, 0.0f)
    , mFrontDir(0.0f, 0.0f, -1.0f)
    , mUpDir(0.0f, 1.0f, 0.0f)
    , mRightDir(1.0f, 0.0f, 0.0f)
    , mViewMatrix()
    , mProjectionMatrix()
    , mFrustumPlaneList()
    , mIsFrustumValid(false)
{
    invalidateFrustum();
    SYRINX_ENSURE(!mIsFrustumValid);
}


void Frustum::setFOVy(float FOVy)
{
    mFOVy = FOVy;
    SYRINX_ENSURE(mFOVy > 0.0 && mFOVy < 180.0);
    invalidateFrustum();
}


void Frustum::setAspectRatio(float aspectRatio)
{
    mAspectRatio = aspectRatio;
    SYRINX_ENSURE(mAspectRatio > 0.0);
    invalidateFrustum();
}


void Frustum::setNearClipDistance(float nearClipDistance)
{
    mNearClipDistance = nearClipDistance;
    SYRINX_ENSURE(mNearClipDistance > 0.0);
    invalidateFrustum();
}


void Frustum::setFarClipDistance(float farClipDistance)
{
    mFarClipDistance = farClipDistance;
    SYRINX_ENSURE(mFarClipDistance > 0.0);
    invalidateFrustum();
}


void Frustum::setPosition(const Point3f& position)
{
    mPosition = position;
    invalidateFrustum();
}


void Frustum::lookAt(const Point3f& lookAt)
{
    Vector3f lookDir = Normalize(lookAt - mPosition);
    setFrontDir(lookDir);
}


void Frustum::setFrontDir(const Vector3f& frontDir)
{
    mFrontDir = Normalize(frontDir);
    invalidateFrustum();
}


const Matrix4x4& Frustum::getViewMatrix()
{
    if (!mIsFrustumValid) {
        updateFrustum();
    }
    return mViewMatrix;
}


const Matrix4x4& Frustum::getProjectionMatrix()
{
    if (!mIsFrustumValid) {
        updateFrustum();
    }
    return mProjectionMatrix;
}


float Frustum::getFOVy() const
{
    return mFOVy;
}


float Frustum::getAspectRation() const
{
    return mAspectRatio;
}


float Frustum::getNearClipDistance() const
{
    return mNearClipDistance;
}


float Frustum::getFarClipDistance() const
{
    return mFarClipDistance;
}


const Point3f& Frustum::getPosition() const
{
    return mPosition;
}


const Vector3f& Frustum::getFrontDir() const
{
    return mFrontDir;
}


bool Frustum::inFrustum(const Point3f& point)
{
    if (!mIsFrustumValid) {
        updateFrustum();
    }

    for (int i = 0; i < FrustumPlane::_size(); ++ i) {
        if (mFrustumPlaneList[i].distanceToPoint(point) < 0.0) {
            return false;
        }
    }
    return true;
}


bool Frustum::inFrustum(const AxisAlignedBox& axisAlignedBox)
{
    if (!mIsFrustumValid) {
        updateFrustum();
    }

    for (int i = 0; i < FrustumPlane::_size(); ++ i) {
        const auto& plane = mFrustumPlaneList[i];

        bool allCornerOutside = true;
        for (size_t cornerIndex = 0; cornerIndex < AxisAlignedBoxCornerIndex::_size() && allCornerOutside; ++ cornerIndex) {
            const Point3f corner = axisAlignedBox.getCornerVertex(AxisAlignedBoxCornerIndex::_from_index(cornerIndex));
            if (plane.distanceToPoint(corner) > 0.0) {
                allCornerOutside = false;
            }
        }

        if (allCornerOutside) {
            return false;
        }
    }
    return true;
}


void Frustum::updateFrustum()
{
    updateViewMatrix();
    updateProjectionMatrix();
    updateFrustumPlaneList();
    mIsFrustumValid = true;
}


void Frustum::updateViewMatrix()
{
    const Vector3f worldUp = Normalize(Vector3f{0.0, 1.0, 0.0});
    mFrontDir = Normalize(mFrontDir);
    mRightDir = Normalize(Cross(mFrontDir, worldUp));
    mUpDir = Normalize(Cross(mRightDir, mFrontDir));
    mViewMatrix = CalculateViewMatrix(mPosition, mPosition + mFrontDir, mUpDir);
}


void Frustum::updateProjectionMatrix()
{
    mProjectionMatrix = CalculateProjectionMatrix(mFOVy, mAspectRatio, mNearClipDistance, mFarClipDistance);
}


void Frustum::updateFrustumPlaneList()
{
    float nearPlaneHeight = mNearClipDistance * std::tan(ConvertDegreeToRadians(mFOVy / 2.0));
    float nearPlaneWidth = nearPlaneHeight * mAspectRatio;

    float farPlaneHeight = mFarClipDistance * std::tan(ConvertDegreeToRadians(mFOVy / 2.0));
    float farPlaneWidth = farPlaneHeight * mAspectRatio;

    const Point3f farPlaneCenter = mPosition + mFrontDir * mFarClipDistance;
    Point3f farPlaneTopLeft     = farPlaneCenter + (mUpDir * (farPlaneHeight / 2.0f)) - (mRightDir * (farPlaneWidth / 2.0f));
    Point3f farPlaneTopRight    = farPlaneCenter + (mUpDir * (farPlaneHeight / 2.0f)) + (mRightDir * (farPlaneWidth / 2.0f));
    Point3f farPlaneBottomLeft  = farPlaneCenter - (mUpDir * (farPlaneHeight / 2.0f)) - (mRightDir * (farPlaneWidth / 2.0f));
    Point3f farPlaneBottomRight = farPlaneCenter - (mUpDir * (farPlaneHeight / 2.0f)) + (mRightDir * (farPlaneWidth / 2.0f));

    const Point3f nearPlaneCenter = mPosition + mFrontDir * mNearClipDistance;
    Point3f nearPlaneTopLeft     = nearPlaneCenter + (mUpDir * (nearPlaneHeight / 2.0f)) - (mRightDir * (nearPlaneWidth / 2.0f));
    Point3f nearPlaneTopRight    = nearPlaneCenter + (mUpDir * (nearPlaneHeight / 2.0f)) + (mRightDir * (nearPlaneWidth / 2.0f));
    Point3f nearPlaneBottomLeft  = nearPlaneCenter - (mUpDir * (nearPlaneHeight / 2.0f)) - (mRightDir * (nearPlaneWidth / 2.0f));
    Point3f nearPlaneBottomRight = nearPlaneCenter - (mUpDir * (nearPlaneHeight / 2.0f)) + (mRightDir * (nearPlaneWidth / 2.0f));

    Plane planeList[FrustumPlane::_size()];

    planeList[FrustumPlane::NearPlane] = Plane(mFrontDir, nearPlaneCenter);
    planeList[FrustumPlane::FarPlane] = Plane(-mFrontDir, farPlaneCenter);

    Vector3f tmpVector;
    tmpVector = (nearPlaneCenter + mRightDir * (nearPlaneWidth / 2.0f)) - mPosition;
    tmpVector = Normalize(tmpVector);
    Normal3f rightPlaneNormal = Normalize(Cross(mUpDir, tmpVector));
    planeList[FrustumPlane::RightPlane] = Plane(rightPlaneNormal, nearPlaneTopRight);

    tmpVector = (nearPlaneCenter - mRightDir * (nearPlaneWidth / 2.0f)) - mPosition;
    tmpVector = Normalize(tmpVector);
    Normal3f leftPlaneNormal = Normalize(Cross(tmpVector, mUpDir));
    planeList[FrustumPlane::LeftPlane] = Plane(leftPlaneNormal, nearPlaneTopLeft);

    tmpVector = (nearPlaneCenter + mUpDir * (nearPlaneHeight / 2.0f)) - mPosition;
    tmpVector = Normalize(tmpVector);
    Normal3f topPlaneNormal = Normalize(Cross(tmpVector, mRightDir));
    planeList[FrustumPlane::TopPlane] = Plane(topPlaneNormal, nearPlaneTopLeft);

    tmpVector = (nearPlaneCenter - mUpDir * (nearPlaneHeight / 2.0f)) - mPosition;
    tmpVector = Normalize(tmpVector);
    Normal3f bottomPlaneNormal = Normalize(Cross(mRightDir, tmpVector));
    planeList[FrustumPlane::BottomPlane] = Plane(bottomPlaneNormal, nearPlaneBottomLeft);

    for (const auto& plane : planeList) {
        mFrustumPlaneList.push_back(plane);
    }
    SYRINX_ENSURE(mFrustumPlaneList.size() == FrustumPlane::_size());
}


void Frustum::invalidateFrustum()
{
    mIsFrustumValid = false;
}

} // namespace Syrinx