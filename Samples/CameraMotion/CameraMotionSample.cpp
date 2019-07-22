#include <Graphics/SyrinxRenderContext.h>
#include <Pipeline/SyrinxEngine.h>
#include <Pipeline/SyrinxRenderPipeline.h>
#include <Pipeline/SyrinxEntityRenderer.h>
#include <Scene/Component/SyrinxCamera.h>
#include <FileSystem/SyrinxFileSystem.h>
#include "CameraController.h"


class LightingPass : public Syrinx::RenderPass {
public:
    LightingPass(const std::string& name) : Syrinx::RenderPass(name) {}

    void onFrameRender(Syrinx::RenderContext& renderContext) override
    {
        renderContext.clearRenderTarget(nullptr, Syrinx::Color(1.0, 0.0, 0.0, 1.0));
        renderContext.clearDepth(nullptr, 1.0);

        Syrinx::EntityRenderer entityRenderer;
        auto cameraEntity = getCamera();
        if (!cameraEntity->hasComponent<Syrinx::Camera>()) {
            return;
        }

        renderContext.pushRenderState();
        renderContext.setRenderState(getRenderState());
        auto& camera = cameraEntity->getComponent<Syrinx::Camera>();
        for (auto entity : getEntityList())
        {
            entityRenderer.render(camera, renderContext, *entity, getShaderName());
        }
        renderContext.popRenderState();
    }
};




int main(int argc, char *argv[])
{
    Syrinx::Engine engine;
    engine.init();
    engine.createWindow("Camera Motion Sample", 800, 800);

    auto fileManager = engine.getFileManager();
    fileManager->addSearchPath("../SampleMedias/");
    fileManager->addSearchPath("../../Medias/Library/");

    auto shaderManager = engine.getShaderManager();
    shaderManager->addShaderSearchPath("../../Medias/Library/");
    shaderManager->addShaderSearchPath("../../Medias/Library/Shader/Unlit");

    Syrinx::Scene *scene = nullptr;
    Syrinx::SceneManager *sceneManager = engine.getSceneManager();
    scene = sceneManager->importScene("cube-scene.scene");
    engine.setActiveScene(scene);

    auto root = scene->getRoot();
    auto cameraNode = scene->createSceneNode("main camera");
    root->addChild(cameraNode);

    auto cameraEntity = sceneManager->createEntity("camera entity");
    cameraNode->attachEntity(cameraEntity);

    auto& cameraTransform = cameraEntity->getComponent<Syrinx::Transform>();
    cameraTransform.translate({0.0, 0.0, 3.0});

    Syrinx::Camera camera("main camera");
    camera.setViewportRect({0, 0, 800, 800});
    cameraEntity->addCamera(camera);

    auto cameraMotionController = std::make_unique<CameraMotionController>();
    cameraEntity->addController(cameraMotionController.get());
    SYRINX_ASSERT(cameraEntity->hasComponent<Syrinx::Controller*>());

    Syrinx::RenderState renderState;
    renderState.viewportState.viewport.extent = {800, 800};

    auto lightingPass = std::make_unique<LightingPass>("lighting");
    lightingPass->setShaderName("display-normal.shader");
    lightingPass->setCamera(cameraEntity);
    lightingPass->setRenderState(&renderState);

    auto renderPipeline = std::make_unique<Syrinx::RenderPipeline>("display constant color");
    renderPipeline->addRenderPass(lightingPass.get());

    engine.addRenderPipeline(renderPipeline.get());
    engine.setActiveRenderPipeline(renderPipeline.get());
    engine.run();
    return 0;
}