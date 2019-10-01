#pragma once
#include <string>
#include <Math/SyrinxMath.h>
#include "Component/SyrinxTransform.h"
#include "Component/SyrinxFrustum.h"

namespace Syrinx {

class Camera {
public:
    explicit Camera(const std::string& name);
    ~Camera() = default;
    Camera(const Camera& rhs);
    Camera& operator=(const Camera& rhs);

    std::string getName() const;
    void setTransform(const Transform* transform);
    void setViewportRect(const ViewportRect& viewportRect);
    Matrix4x4 getViewMatrix() const;
    Matrix4x4 getProjectionMatrix() const;

private:
    std::string mName;
    Frustum mFrustum;
    const Transform *mTransform;
};

} // namespace Syrinx