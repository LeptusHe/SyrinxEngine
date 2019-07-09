#include <Pipeline/SyrinxDisplayDevice.h>
#include <Logging/SyrinxLogManager.h>

int main(int argc, char *argv[])
{
    auto logManager = std::make_unique<Syrinx::LogManager>();
    Syrinx::DisplayDevice displayDevice;

    displayDevice.setMajorVersionNumber(4);
    displayDevice.setMinorVersionNumber(6);
    displayDevice.createWindow("Create Window Sample", 800, 400);

    return 0;
}