#include "Scene/SyrinxScene.h"
#include <Common/SyrinxAssert.h>
#include <Logging/SyrinxLogManager.h>

namespace Syrinx {

Scene::Scene(const std::string& name)
    : Resource(name)
    , mRoot(nullptr)
{
    SYRINX_ENSURE(!mRoot);
}


SceneNode* Scene::createRoot(const std::string& name)
{
    SYRINX_EXPECT(!mRoot);
    mRoot = createSceneNode(name);
    mRoot->setParent(nullptr);
    SYRINX_ENSURE(mRoot);
    return mRoot;
}


SceneNode* Scene::createSceneNode(const std::string& name)
{
    auto* sceneNode = new SceneNode(name, this);
    addSceneNode(sceneNode);
    SYRINX_ENSURE(findSceneNode(name) == sceneNode);
    return sceneNode;
}


SceneNode* Scene::getRoot()
{
    return mRoot;
}


const SceneNode* Scene::getRoot() const
{
    return mRoot;
}


SceneNode* Scene::findSceneNode(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    auto iter = mSceneNodeMap.find(name);
    if (iter == std::end(mSceneNodeMap)) {
        return nullptr;
    }
    return iter->second.get();
}


void Scene::addSceneNode(SceneNode *sceneNode)
{
    SYRINX_EXPECT(sceneNode);
    SYRINX_EXPECT(!findSceneNode(sceneNode->getName()));
    mSceneNodeList.push_back(sceneNode);
    mSceneNodeMap[sceneNode->getName()] = std::unique_ptr<SceneNode>(sceneNode);
    SYRINX_ENSURE(findSceneNode(sceneNode->getName()) == sceneNode);
}


std::vector<Entity*> Scene::getEntityList() const
{
    std::vector<Entity*> entityList;
    for (auto sceneNode : mSceneNodeList) {
        if (auto entity = sceneNode->getEntity(); entity) {
            entityList.push_back(entity);
        }
    }
    return entityList;
}


size_t Scene::getNumSceneNode() const
{
    return mSceneNodeList.size();
}

} // namespace Syrinx