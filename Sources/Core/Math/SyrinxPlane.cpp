#include "Math/SyrinxPlane.h"
#include "Common/SyrinxAssert.h"

namespace Syrinx {

Plane::Plane() : mA(0.0), mB(1.0), mC(0.0), mD(0.0)
{

}


Plane::Plane(float A, float B, float C, float D) : mA(A), mB(B), mC(C), mD(D)
{

}


Plane::Plane(const Normal3f& normal, const Point3f& point)
{
    Normal3f planeNormal = Normalize(normal);
    mA = planeNormal.x;
    mB = planeNormal.y;
    mC = planeNormal.z;
    mD = -(mA * point.x + mB * point.y + mC * point.z);
}


float Plane::distanceToPoint(const Point3f& point) const
{
    return point.x * mA + point.y * mB + point.z * mC + mD;
}

} // namespace Syrinx
