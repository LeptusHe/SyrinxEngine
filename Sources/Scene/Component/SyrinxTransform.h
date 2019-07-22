#pragma once
#include <better-enums/enum.h>
#include <Math/SyrinxMath.h>

namespace Syrinx {

BETTER_ENUM(Space, uint8_t, LocalSpace, WorldSpace);


class Transform {
public:
    Transform();
    Transform(const Transform& rhs);
    explicit Transform(Transform *parent);
    ~Transform() = default;
    Transform& operator=(const Transform& rhs);

    void translate(const Vector3f& translation, Space space = Space::LocalSpace);
    void setScale(const Vector3f& scale);
    void setEulerAngle(const Vector3f& eulerAngle);
    void setWorldMatrix(const Matrix4x4& worldMatrix);
    void combineParentWorldMatrix(const Matrix4x4& parentWorldMatrix);
    const Position& getLocalPosition() const;
    const Vector3f& getScale() const;
    const Vector3f& getEulerAngle() const;
    Matrix4x4 getRotateMatrix() const;
    const Matrix4x4& getLocalMatrix() const;
    const Matrix4x4& getWorldMatrix() const;
    Transform* getParent() const;
    void needUpdate(bool needUpdate);
    bool needUpdate() const;

private:
    Position mLocalPosition;
    Vector3f mScale;
    Vector3f mEulerAngle;
    mutable Matrix4x4 mLocalMatrix;
    mutable Matrix4x4 mParentWorldMatrix;
    Matrix4x4 mWorldMatrix;
    bool mNeedUpdate;
    Transform *mParent;
};

} // namespace Syrinx
