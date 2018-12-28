#include <gmock/gmock.h>
#include <Math/SyrinxColor.h>

using namespace testing;
using namespace Syrinx;


TEST(Color, convert_to_float_array)
{
    float values[] = {1.0, 2.0, 3.0, 4.0};
    Color color(values);
    float *colorValue = color;

    ASSERT_THAT(colorValue, NotNull());
    for (int i = 0; i < 4; ++ i) {
        ASSERT_FLOAT_EQ(values[i], colorValue[i]);
    }
}
