#include "Scene/SyrinxSceneLoader.h"
#include <pugixml.hpp>
#include <Common/SyrinxAssert.h>
#include <Component/SyrinxTransform.h>
#include <Component/SyrinxRenderer.h>
#include <Exception/SyrinxException.h>
#include <ResourceLoader/SyrinxXmlParser.h>
#include "Scene/SyrinxSceneManager.h"

namespace Syrinx {

SceneLoader::SceneLoader(SceneManager *sceneManager, ModelManager *modelManager)
    : mSceneManager(sceneManager)
    , mModelManager(modelManager)
{
    SYRINX_ENSURE(mSceneManager);
    SYRINX_ENSURE(mModelManager);
}


Scene* SceneLoader::loadScene(DataStream& dataStream)
{
    SYRINX_EXPECT(dataStream.isReadable());
    pugi::xml_document sceneDocument;

    Scene *scene = nullptr;
    try {
        sceneDocument.load_string(dataStream.getAsString().c_str());
        auto sceneElement = getChild(sceneDocument, "scene");
        scene = mSceneManager->createScene(dataStream.getName());
        SceneNode *rootNode = scene->createRoot("root");

        for (const auto& nodeElement : sceneElement) {
            if (std::string(nodeElement.name()) != "node") {
                SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "the element [{}] is invalid", nodeElement.name());
            }
            auto childNode = processNode(nodeElement, scene);
            childNode->setParent(rootNode);
            rootNode->addChild(childNode);
        }

    } catch (std::exception& e) {
        delete scene;
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "fail to load scene from stream [{}] because [{}]", dataStream.getName(), e.what());
    }
    return scene;
}


SceneNode* SceneLoader::processNode(const pugi::xml_node& nodeElement, Scene *scene)
{
    std::string name = getAttribute(nodeElement, "name").as_string();
    auto sceneNode = scene->createSceneNode(name);

    auto entityElement = getChild(nodeElement, "entity");
    std::string entityName = getAttribute(entityElement, "name").as_string();
    std::string modelFile = getAttribute(entityElement, "model-file").as_string();
    auto modelEntity = mSceneManager->createEntity(entityName);

    modelEntity->addComponent<Transform>();
    auto& modelTransform = modelEntity->getComponent<Transform>();

    auto positionElement = getChild(nodeElement, "position");
    float xPos = getAttribute(positionElement, "x").as_float();
    float yPos = getAttribute(positionElement, "y").as_float();
    float zPos = getAttribute(positionElement, "z").as_float();
    modelTransform.translate({xPos, yPos, zPos});

    auto scaleElement = getChild(nodeElement, "scale");
    float xScale = getAttribute(scaleElement, "x").as_float();
    float yScale = getAttribute(scaleElement, "y").as_float();
    float zScale = getAttribute(scaleElement, "z").as_float();
    modelTransform.setScale({xScale, yScale, zScale});

    Model* model = mModelManager->createOrRetrieve(modelFile);
    SYRINX_ASSERT(model);
    for (const auto [mesh, material] : model->getMeshMaterialPairList()) {
        auto meshNode = scene->createSceneNode(mesh->getName());
        auto meshEntity = mSceneManager->createEntity(mesh->getName());
        meshEntity->addComponent<Transform>(&modelTransform);
        meshEntity->addComponent<Renderer>();
        auto& meshRenderer = meshEntity->getComponent<Renderer>();
        meshRenderer.setMesh(mesh);
        meshRenderer.setMaterial(material);
        meshNode->attachEntity(meshEntity);

        sceneNode->addChild(meshNode);
        meshNode->setParent(sceneNode);
    }
    sceneNode->attachEntity(modelEntity);
    return sceneNode;
}

} // namespace Syrinx