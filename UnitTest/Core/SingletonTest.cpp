#include <gmock/gmock.h>
#include <Common/SyrinxSingleton.h>

using namespace testing;
using namespace Syrinx;


class Stub : public Singleton<Stub> { };


TEST(Singleton, get_instance_after_create)
{
    Stub *instance = new Stub();
    ASSERT_THAT(Stub::getInstancePtr(), NotNull());
}


TEST(Singleton, assert_failure_to_get_instance_before_create)
{
    ASSERT_DEBUG_DEATH(Stub::getInstancePtr(), "Assertion failed");
    ASSERT_DEBUG_DEATH(Stub::getInstance(), "Assertion failed");
}