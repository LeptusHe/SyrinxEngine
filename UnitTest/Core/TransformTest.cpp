#include <gmock/gmock.h>
#include <Math/SyrinxMath.h>
#include <Component/SyrinxTransform.h>

using namespace testing;
using namespace Syrinx;


TEST(Transform, default_local_position_is_zero)
{
    Transform transform;

    const auto& actualLocalPosition = transform.getLocalPosition();
    Vector3f expectedLocalPosition{0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 3; ++ i) {
        ASSERT_THAT(actualLocalPosition[i], expectedLocalPosition[i]);
    }
}


TEST(Transform, default_local_scale_is_zero)
{
    Transform transform;

    const auto& actualLocalScale = transform.getScale();
    Vector3f expectedLocalScale{1.0f, 1.0f, 1.0f};
    for (int i = 0; i < 3; ++ i) {
        ASSERT_THAT(actualLocalScale[i], expectedLocalScale[i]);
    }
}
