#include <gmock/gmock.h>
#include <HardwareResource/SyrinxHardwareVertexBuffer.h>
#include <HardwareResource/SyrinxHardwareBuffer.h>

using namespace testing;
using namespace Syrinx;


TEST(HardwareVertexBuffer, move_buffer_into_constructor)
{
    auto buffer = std::make_unique<HardwareBuffer>("buffer");
    HardwareVertexBuffer hardwareVertexBuffer(std::move(buffer));

    ASSERT_THAT(buffer, IsNull());
}


TEST(HardwareVertexBuffer, default_vertex_number_is_zero)
{
    auto buffer = std::make_unique<HardwareBuffer>("buffer");
    HardwareVertexBuffer hardwareVertexBuffer(std::move(buffer));

    ASSERT_THAT(hardwareVertexBuffer.getVertexNumber(), 0);
}


TEST(HardwareVertexBuffer, default_vertex_size_is_zero)
{
    auto buffer = std::make_unique<HardwareBuffer>("buffer");
    HardwareVertexBuffer hardwareVertexBuffer(std::move(buffer));

    ASSERT_THAT(hardwareVertexBuffer.getVertexSizeInBytes(), 0);
}