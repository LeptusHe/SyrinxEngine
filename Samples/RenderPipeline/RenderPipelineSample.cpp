#include <RenderPipeline/SyrinxEngine.h>
#include <FileSystem/SyrinxFileSystem.h>

int main(int argc, char *argv[])
{
    Syrinx::Engine engine;
    engine.init();
    engine.createWindow("Render Pipeline Sample", 800, 800);

    auto fileManager = engine.getFileManager();
    fileManager->addSearchPath("../../Medias/");

    Syrinx::Scene *scene = nullptr;
    Syrinx::SceneManager *sceneManager = engine.getSceneManager();
    scene = sceneManager->loadScene("cube-scene.scene");
    engine.setActiveScene(scene);
    auto camera = std::make_unique<Syrinx::Camera>("main camera");
    camera->setPosition({0.0, 8.0, 40.0});
    camera->lookAt({0.0, 8.0, 0.0});
    camera->setViewportRect({0, 0, 800, 800});

    auto lightingPass = std::make_unique<Syrinx::RenderPass>("lighting");
    lightingPass->setShaderPassName("lighting");
    lightingPass->setCamera(camera.get());
    lightingPass->addEntityList(scene->getEntityList());

    auto renderPipeline = std::make_unique<Syrinx::RenderPipeline>("display constant color");
    renderPipeline->addRenderPass(lightingPass.get());

    engine.addRenderPipeline(renderPipeline.get());
    engine.setActiveRenderPipeline(renderPipeline.get());
    engine.run();
    return 0;
}
