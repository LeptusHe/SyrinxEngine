#pragma once
#include <memory>
#include <vector>
#include <unordered_map>
#include <entityx/entityx.h>
#include <ResourceManager/SyrinxModelManager.h>
#include <ResourceManager/SyrinxMaterialManager.h>
#include "System/SyrinxControllerSystem.h"
#include "Scene/SyrinxScene.h"
#include "Scene/SyrinxEntity.h"

namespace Syrinx {

class SceneManager {
public:
    using SceneList = std::vector<Scene*>;
    using SceneMap = std::unordered_map<std::string, std::unique_ptr<Scene>>;
    using EventManager = entityx::EventManager;
    using EntityManager = entityx::EntityManager;
    using SystemManager = entityx::SystemManager;

public:
    SceneManager(FileManager *fileManager, ModelManager *modelManager);
    ~SceneManager() = default;
    Scene* importScene(const std::string& fileName) noexcept(false);
    Scene* createScene(const std::string& name);
    Entity* createEntity(const std::string& name);
    Scene* findScene(const std::string& name) const;
    Entity* findEntity(const std::string& name) const;
    void updateScene(Scene *scene) const;
    void updateController(float timeDelta);

private:
    void addScene(Scene *scene);
    void addEntity(Entity *entity);
    void updateSceneNode(SceneNode *sceneNode, bool parentNeedUpdate) const;
    bool containTransformComponent(SceneNode *sceneNode) const;
    bool needUpdate(SceneNode* sceneNode) const;

private:
    SceneList mSceneList;
    SceneMap mSceneMap;
    FileManager *mFileManager;
    ModelManager *mModelManager;
    EventManager mEventManager;
    EntityManager mEntityManager;
    SystemManager mSystemManager;
};

} // namespace Syrinx