#include <gmock/gmock.h>
#include <HardwareResource/SyrinxHardwareBuffer.h>

using namespace testing;
using namespace Syrinx;


TEST(HardwareBuffer, default_size_is_zero)
{
    HardwareBuffer hardwareBuffer("buffer");
    ASSERT_THAT(hardwareBuffer.getSize(), Eq(0));
}


TEST(HardwareBuffer, default_data_is_nullptr)
{
    HardwareBuffer hardwareBuffer("buffer");
    ASSERT_THAT(hardwareBuffer.getData(), IsNull());
}


TEST(HardwareBuffer, set_data_to_buffer_by_passing_pointer)
{
    HardwareBuffer hardwareBuffer("buffer");

    std::vector<uint8_t> array = {1, 2, 3, 4};
    hardwareBuffer.setSize(array.size() * sizeof(uint8_t));
    hardwareBuffer.setData(array.data());

    auto bufferData = hardwareBuffer.getData();
    for (int i = 0; i < array.size(); ++ i) {
        ASSERT_THAT(bufferData[i], array[i]);
    }
}


TEST(HardwareBuffer, set_data_to_buffer_by_passing_array_pointer)
{
    HardwareBuffer hardwareBuffer("buffer");

    std::vector<uint8_t> array = {1, 2, 3, 4};
    hardwareBuffer.setSize(array.size() * sizeof(uint8_t));
    hardwareBuffer.setData(array.data(), array.size());

    auto bufferData = hardwareBuffer.getData();
    for (int i = 0; i < array.size(); ++ i) {
        ASSERT_THAT(bufferData[i], array[i]);
    }
}