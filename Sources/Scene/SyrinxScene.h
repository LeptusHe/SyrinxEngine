#pragma once
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <RenderResource/SyrinxRenderResource.h>
#include "Scene/SyrinxSceneNode.h"

namespace Syrinx {

class Scene : public RenderResource {
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
    size_t getNumSceneNode() const;

private:
    void addSceneNode(SceneNode *sceneNode);

private:
    SceneNode *mRoot;
    SceneNodeMap mSceneNodeMap;
    SceneNodeList mSceneNodeList;
};

} // namespace Syrinx
