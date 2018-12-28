#include <gmock/gmock.h>
#include <HardwareResource/SyrinxVertexInputState.h>

using namespace testing;
using namespace Syrinx;


TEST(VertexInputState, default_map_size_is_zero)
{
    VertexInputState vertexInputState("vertex input state");

    ASSERT_THAT(vertexInputState.getVertexDataDescriptionMap().size(), Eq(0));
    ASSERT_THAT(vertexInputState.getVertexAttributeDescriptionMap().size(), Eq(0));
}