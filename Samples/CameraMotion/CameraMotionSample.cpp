#include <RenderPipeline/SyrinxEngine.h>
#include <FileSystem/SyrinxFileSystem.h>
#include "CameraController.h"


int main(int argc, char *argv[])
{
    Syrinx::Engine engine;
    engine.init();
    engine.createWindow("Camera Motion Sample", 800, 800);

    auto fileManager = engine.getFileManager();
    fileManager->addSearchPath("../../Medias/");

    Syrinx::Scene *scene = nullptr;
    Syrinx::SceneManager *sceneManager = engine.getSceneManager();
    scene = sceneManager->loadScene("cube-scene.scene");
    engine.setActiveScene(scene);

    auto root = scene->getRoot();
    auto cameraNode = scene->createSceneNode("main camera");
    root->addChild(cameraNode);

    auto cameraEntity = sceneManager->createEntity("camera entity");
    cameraNode->attachEntity(cameraEntity);

    Syrinx::Camera camera("main camera");
    camera.setPosition({0.0, 8.0, 40.0});
    camera.lookAt({0.0, 8.0, 0.0});
    camera.setViewportRect({0, 0, 800, 800});
    cameraEntity->addComponent<Syrinx::Camera>(camera);

    Syrinx::Transform cameraTransform;
    cameraEntity->addComponent<Syrinx::Transform>(cameraTransform);

    auto cameraMotionController = std::make_unique<CameraMotionController>();
    cameraEntity->addController(cameraMotionController.get());
    SYRINX_ASSERT(cameraEntity->hasComponent<Syrinx::Controller*>());

    auto lightingPass = std::make_unique<Syrinx::RenderPass>("lighting");
    lightingPass->setShaderPassName("lighting");
    lightingPass->setCamera(cameraEntity);
    lightingPass->addEntityList(scene->getEntityList());

    auto renderPipeline = std::make_unique<Syrinx::RenderPipeline>("display constant color");
    renderPipeline->addRenderPass(lightingPass.get());

    engine.addRenderPipeline(renderPipeline.get());
    engine.setActiveRenderPipeline(renderPipeline.get());
    engine.run();
    return 0;
}