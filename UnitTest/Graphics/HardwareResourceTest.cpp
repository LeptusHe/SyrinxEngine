#include <gmock/gmock.h>
#include <HardwareResource/SyrinxHardwareResource.h>

using namespace testing;
using namespace Syrinx;


class HardwareResourceMock : public HardwareResource {
public:
    HardwareResourceMock(const std::string& name) : HardwareResource(name) { }

    MOCK_METHOD0(create, bool());
    MOCK_CONST_METHOD0(isValidToCreate, bool());
};


TEST(HardwareResource, state_is_uncreated_after_construct)
{
    HardwareResourceMock hardwareResource("resource");
    ASSERT_THAT(hardwareResource.getState()._value, Eq(HardwareResourceState::Uncreated));
    ASSERT_FALSE(hardwareResource.isCreated());
}
