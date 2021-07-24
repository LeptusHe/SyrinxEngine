#include "SyrinxEngine.h"

namespace Syrinx::Radiance {

Engine::Engine()
    : mTimer(new Timer())
{
    RADIANCE_ENSURE(mTimer);
}

void Engine::setSetting(const DeviceSetting& setting)
{
    mSetting = setting;
}


void Engine::init()
{
    mDisplayDevice = std::make_unique<DisplayDevice>();
    mFileManager = std::make_unique<FileManager>();
    mOptixContext = std::make_unique<OptixContext>();
    mOptixResourceManager = std::make_unique<OptixResourceManager>(mOptixContext.get(), mFileManager.get());
}


RenderWindow * Engine::createWindow(const std::string& title, unsigned int width, unsigned int height)
{
    RADIANCE_EXPECT(mDisplayDevice);
    RADIANCE_EXPECT(getFileManager());

    mDisplayDevice->setMajorVersionNumber(mSetting.getMajorVersionNumber());
    mDisplayDevice->setMinorVersionNumber(mSetting.getMinorVersionNumber());
    mDisplayDevice->setDebugMessageHandler(mSetting.getDebugMessageHandler());
    auto renderWindow = mDisplayDevice->createWindow(title, width, height);
    if (!renderWindow) {
        RADIANCE_THROW_EXCEPTION(ExceptionCode::InvalidState, "fail to create window");
    }
    RADIANCE_ASSERT(renderWindow);
    initInputDevice(renderWindow);
    return renderWindow;
}


FileManager* Engine::getFileManager() const
{
    return mFileManager.get();
}


void Engine::update(float timeDelta)
{
    dispatchEvents();
}


void Engine::run()
{
    auto renderWindow = mDisplayDevice->getRenderWindow();
    while (!shouldStop()) {
        update(mTimer->end());
        renderWindow->swapBuffer();
    }
}


void Engine::initInputDevice(RenderWindow *renderWindow)
{
    RADIANCE_EXPECT(renderWindow);
    mInput = std::make_unique<Input>(renderWindow->fetchWindowHandle());
}


bool Engine::shouldStop() const
{
    return false;
}


void Engine::dispatchEvents()
{
    auto renderWindow = mDisplayDevice->fetchRenderWindow();
    renderWindow->dispatchEvents();
}

} // namespace Syrinx::Radiance