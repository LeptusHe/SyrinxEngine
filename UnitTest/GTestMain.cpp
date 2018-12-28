#include <gmock/gmock.h>
#include <Logging/SyrinxLogManager.h>

int main(int argc, char *argv[])
{
    auto logger = new Syrinx::LogManager();

    testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}