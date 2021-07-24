#pragma once
#include <memory>
#include <Input/LumiereInput.h>
#include <Time/LumiereTimer.h>
#include "Window/RenderWindow.h"
#include "Window/DisplayDevice.h"
#include "Resource/OptixContext.h"
#include "Resource/OptixResourceManager.h"

namespace Syrinx::Radiance {

class Engine {
public:
    Engine();
    ~Engine() = default;

    void setSetting(const DeviceSetting& setting);
    void init();
    RenderWindow* createWindow(const std::string& title, unsigned int width, unsigned int height);
    RenderWindow* getWindow();
    FileManager* getFileManager() const;
    OptixResourceManager* getOptixResourceManager() const;
    void update(float timeDelta);
    void run();

private:
    void initInputDevice(RenderWindow *renderWindow);
    bool shouldStop() const;
    void dispatchEvents();

private:
    DeviceSetting mSetting;
    std::unique_ptr<Timer> mTimer;
    std::unique_ptr<Input> mInput;
    std::unique_ptr<DisplayDevice> mDisplayDevice;
    std::unique_ptr<OptixContext> mOptixContext;
    std::unique_ptr<FileManager> mFileManager;
    std::unique_ptr<OptixResourceManager> mOptixResourceManager;
};

} // namespace Syrinx::Radiance
