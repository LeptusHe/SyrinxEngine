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
    , mEulerAngle(0.0f, 0.0f, 0.0f)
    , mLocalMatrix(1.0f)
    , mWorldMatrix(1.0f)
    , mNeedUpdate(true)
    , mParent(parent)
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


void Transform::setEulerAngle(const Vector3f& eulerAngle)
{
    mEulerAngle = eulerAngle;
}


void Transform::setWorldMatrix(const Matrix4x4& worldMatrix)
{
    mWorldMatrix = worldMatrix;
}


void Transform::combineParentWorldMatrix(const Matrix4x4& parentWorldMatrix)
{
    mWorldMatrix = parentWorldMatrix * getLocalMatrix();
}


const Position& Transform::getLocalPosition() const
{
    return mLocalPosition;
}


const Vector3f& Transform::getScale() const
{
    return mScale;
}


const Vector3f& Transform::getEulerAngle() const
{
    return mEulerAngle;
}


Matrix4x4 Transform::getRotateMatrix() const
{
    glm::mat4 rotateMatrix = glm::mat4(1.0);
    rotateMatrix = glm::rotate(rotateMatrix, glm::radians(mEulerAngle.x), glm::vec3(1.0, 0.0, 0.0));
    rotateMatrix = glm::rotate(rotateMatrix, glm::radians(mEulerAngle.y), glm::vec3(0.0, 1.0, 0.0));
    rotateMatrix = glm::rotate(rotateMatrix, glm::radians(mEulerAngle.z), glm::vec3(0.0, 0.0, 1.0));
    return rotateMatrix;
}


const Matrix4x4& Transform::getLocalMatrix() const
{
    mLocalMatrix = Matrix4x4(1.0f);
    mLocalMatrix = glm::translate(mLocalMatrix, mLocalPosition);
    mLocalMatrix = mLocalMatrix * getRotateMatrix();
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


void Transform::needUpdate(bool needUpdate)
{
    mNeedUpdate = needUpdate;
}


bool Transform::needUpdate() const
{
    return mNeedUpdate;
}

} // namespace Syrinx
