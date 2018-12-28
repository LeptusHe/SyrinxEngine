#include <RenderPipeline/SyrinxDisplayDevice.h>
#include <Logging/SyrinxLogManager.h>

int main(int argc, char *argv[])
{
    Syrinx::LogManager *logManager = new Syrinx::LogManager();
    Syrinx::DisplayDevice displayDevice;

    displayDevice.setMajorVersionNumber(4);
    displayDevice.setMinorVersionNumber(5);
    displayDevice.createWindow("Create Window Sample", 800, 400);

    return 0;
}