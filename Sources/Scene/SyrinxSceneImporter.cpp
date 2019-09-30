#include "Scene/SyrinxSceneImporter.h"
#include <pugixml.hpp>
#include <Common/SyrinxAssert.h>
#include <Component/SyrinxRenderer.h>
#include <Exception/SyrinxException.h>
#include <ResourceLoader/SyrinxXmlParser.h>
#include "Scene/SyrinxSceneManager.h"

namespace Syrinx {

SceneImporter::SceneImporter(SceneManager *sceneManager, ModelManager *modelManager)
    : mSceneManager(sceneManager)
    , mModelManager(modelManager)
{
    SYRINX_ENSURE(mSceneManager);
    SYRINX_ENSURE(mModelManager);
}


Scene* SceneImporter::import(DataStream& dataStream)
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
            auto childNode = processSceneNode(nodeElement, scene);
            childNode->setParent(rootNode);
            rootNode->addChild(childNode);
        }

    } catch (std::exception& e) {
        delete scene;
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "fail to load scene from stream [{}] because [{}]", dataStream.getName(), e.what());
    }
    return scene;
}


SceneNode* SceneImporter::processSceneNode(const pugi::xml_node& nodeElement, Scene *scene)
{
    std::string name = getAttribute(nodeElement, "name").as_string();
    auto sceneNode = scene->createSceneNode(name);

    auto entityElement = getChild(nodeElement, "entity");
    std::string entityName = getAttribute(entityElement, "name").as_string();
    auto entity = mSceneManager->createEntity(entityName);

    Transform transform = processTransform(nodeElement);
    auto& transformComponent = entity->getComponent<Transform>();
    transformComponent = transform;

    std::string entityType = getAttribute(entityElement, "type").as_string();
    if (entityType == "model") {
        processModelEntity(entityElement, scene, sceneNode, transform, entity);
    } else if (entityType == "camera") {
        processCameraEntity(entityElement, scene, sceneNode, transform, entity);
    } else {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "invalid entity type [{}]", entityType);
    }
    sceneNode->attachEntity(entity);

    return sceneNode;
}


Transform SceneImporter::processTransform(const pugi::xml_node& nodeElement)
{
    Transform transform;

    auto positionElement = nodeElement.child("position");
    if (!positionElement.empty()) {
        float xPos = getAttribute(positionElement, "x").as_float();
        float yPos = getAttribute(positionElement, "y").as_float();
        float zPos = getAttribute(positionElement, "z").as_float();
        transform.translate({xPos, yPos, zPos});
    }

    auto scaleElement = nodeElement.child("scale");
    if (!scaleElement.empty()) {
        float xScale = getAttribute(scaleElement, "x").as_float();
        float yScale = getAttribute(scaleElement, "y").as_float();
        float zScale = getAttribute(scaleElement, "z").as_float();
        transform.setScale({xScale, yScale, zScale});
    }

    return transform;
}


void SceneImporter::processModelEntity(const pugi::xml_node& entityElement, Scene *scene, SceneNode *sceneNode, const Transform& transform, Entity *entity)
{
    SYRINX_EXPECT(scene && sceneNode && entity);
    std::string modelFile = getAttribute(entityElement, "model-file").as_string();
    Model* model = mModelManager->createOrRetrieve(modelFile);
    SYRINX_ASSERT(model);
    for (const auto [mesh, material] : model->getMeshMaterialPairList()) {
        auto meshNode = scene->createSceneNode(mesh->getName());
        auto meshEntity = mSceneManager->createEntity(mesh->getName());

        auto& transformComponent = meshEntity->getComponent<Transform>();
        transformComponent = transform;
        meshEntity->addComponent<Renderer>();
        auto& meshRenderer = meshEntity->getComponent<Renderer>();
        meshRenderer.setMesh(mesh);
        meshRenderer.setMaterial(material);
        meshNode->attachEntity(meshEntity);

        sceneNode->addChild(meshNode);
        meshNode->setParent(sceneNode);
    }
}


void SceneImporter::processCameraEntity(const pugi::xml_node& entityElement, Scene *scene, SceneNode *sceneNode, const Transform& transform, Entity *entity)
{
    SYRINX_EXPECT(scene && sceneNode && entity);
    auto entityName = getAttribute(entityElement, "name").as_string();
    Camera camera(entityName);
    entity->addCamera(camera);
}

} // namespace Syrinx