#include "Scene/SyrinxSceneNode.h"
#include <Common/SyrinxAssert.h>
#include "Scene/SyrinxScene.h"

namespace Syrinx {

SceneNode::SceneNode(const std::string& name, Scene *scene)
    : mName(name)
    , mParent(nullptr)
    , mChildren()
    , mEntity(nullptr)
    , mScene(scene)
{
    SYRINX_ENSURE(!mName.empty());
    SYRINX_ENSURE(!mParent);
    SYRINX_ENSURE(mChildren.empty());
    SYRINX_ENSURE(!mEntity);
    SYRINX_ENSURE(mScene);
    SYRINX_ENSURE(mScene == scene);
}


void SceneNode::setParent(SceneNode *sceneNode)
{
    SYRINX_EXPECT(sceneNode ? (mScene == &sceneNode->getScene()) : true);
    mParent = sceneNode;
}


void SceneNode::addChild(SceneNode *sceneNode)
{
    SYRINX_EXPECT(sceneNode);
    SYRINX_EXPECT(&sceneNode->getScene() == mScene);
    mChildren.push_back(sceneNode);
}


void SceneNode::attachEntity(Entity *entity)
{
    mEntity = entity;
    SYRINX_ENSURE(mEntity);
    SYRINX_ENSURE(mEntity == entity);
}


const std::string& SceneNode::getName() const
{
    return mName;
}


SceneNode* SceneNode::getParent() const
{
    return mParent;
}


size_t SceneNode::getNumChild() const
{
    return mChildren.size();
}


const SceneNode::Children& SceneNode::getChildren() const
{
    return mChildren;
}


SceneNode* SceneNode::getChild(const std::string& name) const
{
    SYRINX_EXPECT(!name.empty());
    for (auto child : mChildren) {
        if (child->getName() == name) {
            return child;
        }
    }
    return nullptr;
}


Entity* SceneNode::getEntity() const
{
    return mEntity;
}


const Scene& SceneNode::getScene() const
{
    SYRINX_EXPECT(mScene);
    return *mScene;
}

} // namespace Syrinx