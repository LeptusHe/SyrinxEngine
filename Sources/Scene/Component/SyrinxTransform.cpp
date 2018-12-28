#include "Component/SyrinxTransform.h"
#include <Common/SyrinxAssert.h>

namespace Syrinx {

Transform::Transform() : Transform(nullptr)
{
    SYRINX_ENSURE(!mParent);
}


Transform::Transform(Transform *parent)
    : mLocalPosition(0.0f, 0.0f, 0.0f)
    , mScale(1.0f, 1.0f, 1.0f)
    , mLocalMatrix(1.0f)
    , mWorldMatrix(1.0f)
    , mParent(parent)
    , mNeedUpdate(true)
{
    SYRINX_ENSURE(mNeedUpdate);
}


void Transform::translate(const Vector3f& translation, Space space)
{
    mLocalPosition += translation;
    mNeedUpdate = true;
}


void Transform::setScale(const Vector3f& scale)
{
    mScale = scale;
    mNeedUpdate = true;
}


void Transform::setWorldMatrix(const Matrix4x4& worldMatrix)
{
    mWorldMatrix = worldMatrix;
}


void Transform::combineParentWorldMatrix(const Matrix4x4& parentWorldMatrix)
{
    mWorldMatrix = parentWorldMatrix * getLocalMatrix();
}


void Transform::needUpdate(bool needUpdate)
{
    mNeedUpdate = needUpdate;
}


const Position& Transform::getLocalPosition() const
{
    return mLocalPosition;
}


const Vector3f& Transform::getScale() const
{
    return mScale;
}


const Matrix4x4& Transform::getLocalMatrix() const
{
    mLocalMatrix = Matrix4x4(1.0f);
    mLocalMatrix = glm::translate(mLocalMatrix, mLocalPosition);
    mLocalMatrix = glm::scale(mLocalMatrix, mScale);
    return mLocalMatrix;
}


const Matrix4x4& Transform::getWorldMatrix() const
{
    return mWorldMatrix;
}


Transform* Transform::getParent() const
{
    return mParent;
}


bool Transform::needUpdate() const
{
    return mNeedUpdate;
}

} // namespace Syrinx
