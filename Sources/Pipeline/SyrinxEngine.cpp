#include "SyrinxEngine.h"
#include <Exception/SyrinxException.h>
#include "SyrinxCommandSubmitter.h"

namespace Syrinx {

Engine::Engine()
    : mSetting()
    , mTimer(new Timer())
    , mInput(nullptr)
    , mDisplayDevice(nullptr)
    , mLogManager(nullptr)
    , mFileManager(nullptr)
    , mHardwareResourceManager(nullptr)
    , mMeshManager(nullptr)
    , mMaterialManager(nullptr)
    , mModelManager(nullptr)
    , mSceneManager(nullptr)
    , mActiveScene(nullptr)
    , mRenderPipelineList()
    , mActiveRenderPipeline(nullptr)
{
    SYRINX_ENSURE(mTimer);
    SYRINX_ENSURE(!mInput);
    SYRINX_ENSURE(!mDisplayDevice);
    SYRINX_ENSURE(!mLogManager);
    SYRINX_ENSURE(!mFileManager);
    SYRINX_ENSURE(!mHardwareResourceManager);
    SYRINX_ENSURE(!mMeshManager);
    SYRINX_ENSURE(!mMaterialManager);
    SYRINX_ENSURE(!mModelManager);
    SYRINX_ENSURE(!mSceneManager);
    SYRINX_ENSURE(!mActiveScene);
    SYRINX_ENSURE(mRenderPipelineList.empty());
    SYRINX_ENSURE(!mActiveRenderPipeline);
}


void Engine::setEngineSetting(const EngineSetting& setting)
{
    mSetting = setting;
    SYRINX_ENSURE(mSetting.isVersionValid());
}


void Engine::init()
{
    mDisplayDevice = std::make_unique<DisplayDevice>();
    mLogManager = std::make_unique<LogManager>();
    mFileManager = std::make_unique<FileManager>();
    mHardwareResourceManager = std::make_unique<HardwareResourceManager>();
    mMeshManager = std::make_unique<MeshManager>(getFileManager(), getHardwareResourceManager());
    mShaderManager = std::make_unique<ShaderManager>(getFileManager(), getHardwareResourceManager());
    mMaterialManager = std::make_unique<MaterialManager>(getFileManager(), getHardwareResourceManager(), getShaderManager());
    mModelManager = std::make_unique<ModelManager>(getFileManager(), getMeshManager(), getMaterialManager());
    mSceneManager = std::make_unique<SceneManager>(getFileManager(), getModelManager());
}


RenderWindow* Engine::createWindow(const std::string& title, unsigned int width, unsigned int height)
{
    SYRINX_EXPECT(mDisplayDevice);
    mDisplayDevice->setMajorVersionNumber(mSetting.getMajorVersionNumber());
    mDisplayDevice->setMinorVersionNumber(mSetting.getMinorVersionNumber());
    mDisplayDevice->setDebugMessageHandler(mSetting.getDebugMessageHandler());
    auto renderWindow = mDisplayDevice->createWindow(title, width, height);
    SYRINX_ASSERT(renderWindow);
    initInputDevice(renderWindow);
    return renderWindow;
}


void Engine::initInputDevice(RenderWindow *renderWindow)
{
    SYRINX_EXPECT(renderWindow);
    mInput = std::make_unique<Input>(renderWindow->fetchWindowHandle());
}


void Engine::addRenderPipeline(RenderPipeline* renderPipeline)
{
    SYRINX_EXPECT(renderPipeline);
    mRenderPipelineList.push_back(renderPipeline);
}


RenderPipeline* Engine::getRenderPipeline(const std::string& name) const
{
    SYRINX_EXPECT(!name.empty());
    for (const auto& renderPipeline : mRenderPipelineList) {
        if (renderPipeline->getName() == name) {
            return renderPipeline;
        }
    }
    return nullptr;
}


void Engine::setActiveRenderPipeline(RenderPipeline *renderPipeline)
{
    SYRINX_EXPECT(renderPipeline);
    SYRINX_EXPECT(getRenderPipeline(renderPipeline->getName()));
    mActiveRenderPipeline = renderPipeline;
    SYRINX_ENSURE(mActiveRenderPipeline == renderPipeline);
}


void Engine::setActiveScene(Scene *scene)
{
    SYRINX_EXPECT(scene);
    mActiveScene = scene;
}


void Engine::update(float timeDelta)
{
    dispatchEvents();
    mSceneManager->updateController(timeDelta);
    mSceneManager->updateScene(mActiveScene);
}


void Engine::run()
{
    SYRINX_EXPECT(isValidToRun());
    auto renderWindow = mDisplayDevice->getRenderWindow();
    CommandSubmitter commandSubmitter(mShaderManager.get());
    while (!shouldStop()) {
        SYRINX_ASSERT(mActiveRenderPipeline);
        update(mTimer->end());
        commandSubmitter.submit(*mActiveRenderPipeline);
        renderWindow->swapBuffer();
        mTimer->start();
    }
}


FileManager* Engine::getFileManager() const
{
    SYRINX_EXPECT(mFileManager);
    return mFileManager.get();
}


HardwareResourceManager* Engine::getHardwareResourceManager() const
{
    SYRINX_EXPECT(mHardwareResourceManager);
    return mHardwareResourceManager.get();
}


MeshManager* Engine::getMeshManager() const
{
    SYRINX_EXPECT(mMeshManager);
    return mMeshManager.get();
}


ShaderManager* Engine::getShaderManager() const
{
    SYRINX_EXPECT(mShaderManager);
    return mShaderManager.get();
}


MaterialManager* Engine::getMaterialManager() const
{
    SYRINX_EXPECT(mMaterialManager);
    return mMaterialManager.get();
}


ModelManager* Engine::getModelManager() const
{
    SYRINX_EXPECT(mModelManager);
    return mModelManager.get();
}


SceneManager* Engine::getSceneManager() const
{
    SYRINX_EXPECT(mSceneManager);
    return mSceneManager.get();
}


bool Engine::shouldStop() const
{
    return false;
}


bool Engine::isValidToRun() const
{
    return mActiveScene && mActiveRenderPipeline;
}


void Engine::dispatchEvents()
{
    auto renderWindow = mDisplayDevice->fetchRenderWindow();
    renderWindow->dispatchEvents();
}

} // namespace Syrinx