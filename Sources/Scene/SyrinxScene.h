#pragma once
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <RenderResource/SyrinxResource.h>
#include "Scene/SyrinxSceneNode.h"

namespace Syrinx {

class Scene : public Resource {
public:
    using SceneNodeList = std::vector<SceneNode*>;
    using SceneNodeMap = std::unordered_map<std::string, std::unique_ptr<SceneNode>>;

public:
    explicit Scene(const std::string& name);
    ~Scene() override = default;

    SceneNode* createRoot(const std::string& name);
    SceneNode* createSceneNode(const std::string& name);
    SceneNode* getRoot();
    const SceneNode* getRoot() const;
    SceneNode* findSceneNode(const std::string& name);
    std::vector<Entity*> getEntityList() const;
    template <typename T> std::vector<Entity*> getEntitiesWithComponent() const;
    size_t getNumSceneNode() const;

private:
    void addSceneNode(SceneNode *sceneNode);

private:
    SceneNode *mRoot;
    SceneNodeMap mSceneNodeMap;
    SceneNodeList mSceneNodeList;
};




template <typename T>
std::vector<Entity*> Scene::getEntitiesWithComponent() const
{
    std::vector<Entity*> entityList;
    for (auto sceneNode : mSceneNodeList) {
        auto entity = sceneNode->getEntity();
        if (entity && entity->hasComponent<T>()) {
            entityList.push_back(entity);
        }
    }
    return entityList;
}

} // namespace Syrinx
