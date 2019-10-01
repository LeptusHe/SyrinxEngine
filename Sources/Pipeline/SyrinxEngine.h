#pragma once
#include <string>
#include <memory>
#include <Common/SyrinxSingleton.h>
#include <Input/SyrinxInput.h>
#include <Logging/SyrinxLogManager.h>
#include <Time/SyrinxTimer.h>
#include <FileSystem/SyrinxFileManager.h>
#include <GUI/SyrinxGui.h>
#include <Graphics/SyrinxRenderContext.h>
#include <Manager/SyrinxHardwareResourceManager.h>
#include <ResourceManager/SyrinxMeshManager.h>
#include <ResourceManager/SyrinxModelManager.h>
#include <ResourceManager/SyrinxShaderManager.h>
#include <ResourceManager/SyrinxMaterialManager.h>
#include <Scene/SyrinxSceneManager.h>
#include "SyrinxDisplayDevice.h"
#include "SyrinxEngineSetting.h"
#include "SyrinxScriptableRenderPipeline.h"

namespace Syrinx {

class Engine : public Singleton<Engine> {
public:
    using ScriptableRenderPipelineList = std::vector<IScriptableRenderPipeline*>;

public:
    Engine();
    ~Engine() = default;

    void setEngineSetting(const EngineSetting& setting);
    void init();
    RenderWindow* createWindow(const std::string& title, unsigned int width, unsigned int height);
    RenderWindow* getWindow();
    void addRenderPipeline(IScriptableRenderPipeline* renderPipeline);
    IScriptableRenderPipeline* getRenderPipeline(const std::string& name) const;
    void setActiveRenderPipeline(IScriptableRenderPipeline* renderPipeline);
    void setActiveScene(Scene *scene);
    IScriptableRenderPipeline* getActiveRenderPipeline() const;
    Scene* getActiveScene() const;
    FileManager* getFileManager() const;
    HardwareResourceManager* getHardwareResourceManager() const;
    MeshManager* getMeshManager() const;
    ShaderManager* getShaderManager() const;
    MaterialManager* getMaterialManager() const;
    ModelManager* getModelManager() const;
    SceneManager* getSceneManager() const;
    void update(float timeDelta);
    void run();

private:
    void initInputDevice(RenderWindow *renderWindow);
    bool shouldStop() const;
    bool isValidToRun() const;
    void dispatchEvents();

private:
    EngineSetting mSetting;
    std::unique_ptr<Timer> mTimer;
    std::unique_ptr<Input> mInput;
    std::unique_ptr<Gui> mGui;
    std::unique_ptr<RenderContext> mRenderContext;
    std::unique_ptr<DisplayDevice> mDisplayDevice;
    std::unique_ptr<LogManager> mLogManager;
    std::unique_ptr<FileManager> mFileManager;
    std::unique_ptr<HardwareResourceManager> mHardwareResourceManager;
    std::unique_ptr<MeshManager> mMeshManager;
    std::unique_ptr<ShaderManager> mShaderManager;
    std::unique_ptr<MaterialManager> mMaterialManager;
    std::unique_ptr<ModelManager> mModelManager;
    std::unique_ptr<SceneManager> mSceneManager;
    Scene *mActiveScene;
    ScriptableRenderPipelineList mRenderPipelineList;
    IScriptableRenderPipeline *mActiveRenderPipeline;
};

} // namespace Syrinx