#pragma once
#include <Streaming/SyrinxDataStream.h>
#include <pugixml.hpp>
#include "ResourceManager/SyrinxModelManager.h"
#include "Scene/SyrinxScene.h"
#include "Scene/SyrinxSceneNode.h"

namespace Syrinx {

class SceneManager;

class SceneLoader {
public:
    SceneLoader(SceneManager *sceneManager, ModelManager *modelManager);
    ~SceneLoader() = default;
    Scene* loadScene(DataStream& dataStream);

private:
    SceneNode* processNode(const pugi::xml_node& nodeElement, Scene *scene);

private:
    SceneManager *mSceneManager;
    ModelManager *mModelManager;
};

} // namespace Syrinx