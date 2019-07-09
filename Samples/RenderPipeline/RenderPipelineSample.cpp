#include <Pipeline/SyrinxEngine.h>
#include <FileSystem/SyrinxFileSystem.h>

int main(int argc, char *argv[])
{
    Syrinx::Engine engine;
    engine.init();
    engine.createWindow("Render Pipeline Sample", 800, 800);

    auto fileManager = engine.getFileManager();
    fileManager->addSearchPath("../SampleMedias/");

    auto shaderManager = engine.getShaderManager();
    shaderManager->addShaderSearchPath("../../Medias/Library/");

    Syrinx::Scene *scene = nullptr;
    Syrinx::SceneManager *sceneManager = engine.getSceneManager();
    scene = sceneManager->importScene("cube-scene.scene");
    engine.setActiveScene(scene);

    auto root = scene->getRoot();
    auto cameraNode = scene->createSceneNode("main camera");
    root->addChild(cameraNode);

    auto cameraEntity = sceneManager->createEntity("camera entity");
    cameraNode->attachEntity(cameraEntity);

    Syrinx::Camera camera("main camera");
    camera.setPosition({0.0, 0.0, 3.0});
    camera.lookAt({0.0, 0.0, 2.0});
    camera.setViewportRect({0, 0, 800, 800});
    cameraEntity->addComponent<Syrinx::Camera>(camera);

    Syrinx::Transform cameraTransform;
    cameraEntity->addComponent<Syrinx::Transform>(cameraTransform);

    Syrinx::RenderState renderState;

    auto lightingPass = std::make_unique<Syrinx::RenderPass>("lighting pass");
    lightingPass->setShaderName("constant-color.shader");
    lightingPass->setCamera(cameraEntity);
    lightingPass->setRenderState(&renderState);
    lightingPass->addEntityList(scene->getEntityList());

    auto renderPipeline = std::make_unique<Syrinx::RenderPipeline>("display constant color");
    renderPipeline->addRenderPass(lightingPass.get());

    engine.addRenderPipeline(renderPipeline.get());
    engine.setActiveRenderPipeline(renderPipeline.get());
    engine.run();

    return 0;
}
