#include "Scene/SyrinxSceneManager.h"
#include <Exception/SyrinxException.h>
#include "Scene/SyrinxSceneImporter.h"
#include "Component/SyrinxTransform.h"
#include "System/SyrinxControllerSystem.h"

namespace Syrinx {

SceneManager::SceneManager(FileManager *fileManager, ModelManager *modelManager)
    : mFileManager(fileManager)
    , mModelManager(modelManager)
    , mEventManager()
    , mEntityManager(mEventManager)
    , mSystemManager(mEntityManager, mEventManager)
{
    SYRINX_ENSURE(mFileManager);
    SYRINX_ENSURE(mModelManager);
    mSystemManager.add<ControllerSystem>();
    mSystemManager.configure();
}


Scene* SceneManager::importScene(const std::string& fileName)
{
    SYRINX_EXPECT(!fileName.empty());
    auto [fileExist, filePath] = mFileManager->findFile(fileName);
    if (!fileExist) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileNotFound, "can not find scene file [{}]", fileName);
    }
    auto dataStream = mFileManager->openFile(filePath, FileAccessMode::READ);
    if (!dataStream) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileSystemError, "fail to open scene file [{}]", filePath);
    }
    SceneImporter sceneImporter(this, mModelManager);
    Scene *scene = sceneImporter.import(*dataStream);
    return scene;
}


Scene* SceneManager::createScene(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    auto scene = new Scene(name);
    addScene(scene);
    return scene;
}


Entity* SceneManager::createEntity(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    auto entityId = mEntityManager.create();
    auto entity = new Entity(name, entityId);
    //auto model = mModelManager->createModel(modelFile);
    //auto material = mMaterialManager->createOrRetrieveMaterial(materialFile);
    //entity->setModel(model);
    //entity->setMaterial(material);
    return entity;
}


Scene* SceneManager::findScene(const std::string& name) const
{
    SYRINX_EXPECT(!name.empty());
    auto iter = mSceneMap.find(name);
    if (iter == std::end(mSceneMap)) {
        return nullptr;
    }
    return iter->second.get();
}


Entity* SceneManager::findEntity(const std::string& name) const
{
    //SYRINX_EXPECT(!modelName.empty());
    return nullptr;
    /*
    auto iter = mEntityMap.find(modelName);
    if (iter != std::end(mEntityMap)) {
        return iter->second;
    }
    return nullptr;
     */
}


void SceneManager::addScene(Scene *scene)
{
    SYRINX_EXPECT(scene);
    SYRINX_EXPECT(!findScene(scene->getName()));
    mSceneList.push_back(scene);
    mSceneMap[scene->getName()] = std::unique_ptr<Scene>(scene);
    SYRINX_ENSURE(findScene(scene->getName()) == scene);
}


void SceneManager::addEntity(Entity *entity)
{
    SYRINX_EXPECT(entity);
    /*
    mEntityList.push_back(std::unique_ptr<Entity>(entity));
    mEntityMap[entity->getName()] = entity;
     */
}


void SceneManager::updateScene(Scene *scene) const
{
    SYRINX_EXPECT(scene);
    updateSceneNode(scene->getRoot(), false);
}


void SceneManager::updateController(float timeDelta)
{
    mSystemManager.update<ControllerSystem>(timeDelta);
}


void SceneManager::updateSceneNode(SceneNode *sceneNode, bool parentNeedUpdate) const
{
    if (!sceneNode) {
        return;
    }

    bool selfNeedUpdate = needUpdate(sceneNode);
    if (containTransformComponent(sceneNode) && (parentNeedUpdate || selfNeedUpdate)) {
        auto& transform = sceneNode->getEntity()->getComponent<Transform>();
        Transform* parent = transform.getParent();
        if (!parent) {
            transform.setWorldMatrix(transform.getLocalMatrix());
        } else {
            const auto& parentWorldMatrix = parent->getWorldMatrix();
            transform.combineParentWorldMatrix(parentWorldMatrix);
        }
        transform.needUpdate(false);
    }

    for (auto childNode : sceneNode->getChildren()) {
        updateSceneNode(childNode, parentNeedUpdate || selfNeedUpdate);
    }
}


bool SceneManager::needUpdate(SceneNode *sceneNode) const
{
    SYRINX_EXPECT(sceneNode);
    auto entity = sceneNode->getEntity();
    if (!containTransformComponent(sceneNode)) {
        return false;
    }
    const auto& transform = entity->getComponent<Transform>();
    return transform.needUpdate();
}


bool SceneManager::containTransformComponent(SceneNode *sceneNode) const
{
    SYRINX_EXPECT(sceneNode);
    auto entity = sceneNode->getEntity();
    return entity && entity->hasComponent<Transform>();
}

} // namespace Syrinx