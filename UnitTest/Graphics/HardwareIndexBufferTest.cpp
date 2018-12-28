#include <gmock/gmock.h>
#include <HardwareResource/SyrinxHardwareIndexBuffer.h>

using namespace testing;
using namespace Syrinx;

TEST(HardwareIndexBuffer, move_buffer_into_constructor)
{
    auto buffer = std::make_unique<HardwareBuffer>("index buffer");
    HardwareIndexBuffer hardwareIndexBuffer(std::move(buffer));

    ASSERT_THAT(buffer, IsNull());
}


TEST(HardwareIndexBuffer, default_index_type_is_uint32)
{
    auto buffer = std::make_unique<HardwareBuffer>("index buffer");
    HardwareIndexBuffer hardwareIndexBuffer(std::move(buffer));

    ASSERT_THAT(hardwareIndexBuffer.getIndexType()._value, Eq(IndexType::UINT32));
}


TEST(HardwareIndexBuffer, default_index_num_is_zero)
{
    auto buffer = std::make_unique<HardwareBuffer>("index buffer");
    HardwareIndexBuffer hardwareIndexBuffer(std::move(buffer));

    ASSERT_THAT(hardwareIndexBuffer.getNumIndexes(), Eq(0));
}
