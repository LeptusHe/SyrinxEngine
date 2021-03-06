#pragma once
#include <better-enums/enum.h>
#include <Math/SyrinxMath.h>

namespace Syrinx {

BETTER_ENUM(Space, uint8_t, LocalSpace, WorldSpace);


class Transform {
public:
    Transform();
    explicit Transform(Transform *parent);
    ~Transform() = default;

    void translate(const Vector3f& translation, Space space = Space::LocalSpace);
    void setScale(const Vector3f& scale);
    void setWorldMatrix(const Matrix4x4& worldMatrix);
    void combineParentWorldMatrix(const Matrix4x4& parentWorldMatrix);
    void needUpdate(bool needUpdate);
    const Position& getLocalPosition() const;
    const Vector3f& getScale() const;
    const Matrix4x4& getLocalMatrix() const;
    const Matrix4x4& getWorldMatrix() const;
    Transform* getParent() const;
    bool needUpdate() const;

private:
    Position mLocalPosition;
    Vector3f mScale;
    mutable Matrix4x4 mLocalMatrix;
    Matrix4x4 mWorldMatrix;
    bool mNeedUpdate;
    Transform *mParent;
};

} // namespace Syrinx
