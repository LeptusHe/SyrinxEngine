#include <gmock/gmock.h>
#include <Math/SyrinxPlane.h>

using namespace testing;
using namespace Syrinx;


TEST(Plane, dist_to_point)
{
    Plane plane(0.0f, 1.0f, 0.0f, -2.0f);
    float dist = plane.distanceToPoint({0.0, 0.0, 0.0});
    ASSERT_THAT(dist, FloatEq(-2.0));
}
