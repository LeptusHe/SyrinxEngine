#pragma once
#include <string>
#include <vector>
#include <Math/SyrinxMath.h>
#include <Scene/SyrinxEntity.h>

namespace Syrinx {

class Scene;
class SceneManager;

class SceneNode {
public:
    using Children = std::vector<SceneNode*>;

public:
    explicit SceneNode(const std::string& name, Scene *scene);
    ~SceneNode() = default;

    void setParent(SceneNode *sceneNode);
    void addChild(SceneNode *sceneNode);
    void attachEntity(Entity *entity);
    const std::string& getName() const;
    SceneNode* getParent() const;
    size_t getNumChild() const;
    const Children& getChildren() const;
    SceneNode* getChild(const std::string& name) const;
    Entity* getEntity() const;
    const Scene& getScene() const;

private:
    std::string mName;
    SceneNode *mParent;
    Children mChildren;
    Entity *mEntity;
    Scene *mScene;
};

} // namespace Syrinx