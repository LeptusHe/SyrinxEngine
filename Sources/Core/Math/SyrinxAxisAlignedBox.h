#pragma once
#include "Math/SyrinxMath.h"
#include <better-enums/enum.h>

namespace Syrinx {

BETTER_ENUM(AxisAlignedBoxCornerIndex, uint8_t,
        FAR_LEFT_BOTTOM,
        FAR_LEFT_TOP,
        FAR_RIGHT_TOP,
        FAR_RIGHT_BOTTOM,
        NEAR_LEFT_BOTTOM,
        NEAR_LEFT_TOP,
        NEAR_RIGHT_TOP,
        NEAR_RIGHT_BOTTOM);


class AxisAlignedBox {
public:
    AxisAlignedBox(const Vector3f& minimum, const Vector3f& maximum);
    const Vector3f& getMinimum() const;
    const Vector3f& getMaximum() const;
    Point3f getCornerVertex(AxisAlignedBoxCornerIndex cornerIndex) const;

private:
    Vector3f mMinimum;
    Vector3f mMaximum;
};

} // namespace Syrinx