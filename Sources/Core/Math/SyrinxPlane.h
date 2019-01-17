#pragma once
#include "Math/SyrinxMath.h"

namespace Syrinx {

class Plane {
public:
    Plane();
    ~Plane() = default;
    Plane(float A, float B, float C, float D);
    Plane(const Normal3f& normal, const Point3f& point);
    float distanceToPoint(const Point3f& point) const;

private:
    float mA;
    float mB;
    float mC;
    float mD;
};

} // namespace Syrinx