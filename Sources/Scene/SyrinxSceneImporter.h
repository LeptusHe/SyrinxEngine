#pragma once
#include <Streaming/SyrinxDataStream.h>
#include <pugixml.hpp>
#include <ResourceManager/SyrinxModelManager.h>
#include "Scene/SyrinxScene.h"
#include "Scene/SyrinxSceneNode.h"

namespace Syrinx {

class SceneManager;

class SceneImporter {
public:
    SceneImporter(SceneManager *sceneManager, ModelManager *modelManager);
    ~SceneImporter() = default;
    Scene* import(DataStream& dataStream);

private:
    SceneNode* processSceneNode(const pugi::xml_node& nodeElement, Scene *scene);
    Transform processTransform(const pugi::xml_node& nodeElement);
    void processModelEntity(const pugi::xml_node& entityElement, Scene *scene, SceneNode *sceneNode, const Transform& transform, Entity *entity);
    void processCameraEntity(const pugi::xml_node& entityElement, Scene *scene, SceneNode *sceneNode, const Transform& transform, Entity *entity);

private:
    SceneManager *mSceneManager;
    ModelManager *mModelManager;
};

} // namespace Syrinx