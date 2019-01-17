#pragma once
#include "RenderPipeline/SyrinxRenderWindow.h"
#include "RenderPipeline/SyrinxEngineSetting.h"

namespace Syrinx {

class DisplayDevice {
public:
    DisplayDevice();
    ~DisplayDevice();
    DisplayDevice(const DisplayDevice&) = delete;
    DisplayDevice& operator=(const DisplayDevice&) = delete;

    RenderWindow* createWindow(const std::string& title, unsigned width, unsigned height);
    const RenderWindow* getRenderWindow() const;
    RenderWindow* fetchRenderWindow();
    void setMajorVersionNumber(unsigned majorVersionNumber);
    void setMinorVersionNumber(unsigned minorVersionNumber);
    void setDebugMessageHandler(DebugMessageHandler debugMessageHandler);
    unsigned getMajorVersionNumber() const;
    unsigned getMinorVersionNumber() const;
    DebugMessageHandler getDebugMessageHandler() const;

private:
    bool initWindow(const std::string& title, unsigned width, unsigned height);
    bool init();
    bool loadFunctions();
    bool initDebugMessageHandler();
    bool isDebugContextCreated() const;
    bool isVersionNumberValid() const;

private:
    unsigned mMajorVersionNumber;
    unsigned mMinorVersionNumber;
    DebugMessageHandler mDebugMessageHandler;
    std::unique_ptr<RenderWindow> mRenderWindow;
};

} // namespace Syrinx
