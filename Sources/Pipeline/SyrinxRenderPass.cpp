#include <Scene/Component/SyrinxRenderer.h>
#include "SyrinxRenderPass.h"

namespace Syrinx {

RenderPass::RenderPass(const std::string& name)
    : mName(name)
    , mShaderName()
    , mCamera(nullptr)
    , mEntityList()
    , mRenderState(nullptr)
{
    SYRINX_ENSURE(!mName.empty());
    SYRINX_ENSURE(mShaderName.empty());
    SYRINX_ENSURE(!mCamera);
    SYRINX_ENSURE(mEntityList.empty());
    SYRINX_ENSURE(!mRenderState);
}


void RenderPass::onInit(Scene *scene)
{
    if (!scene) {
        return;
    }

    for (auto entity : scene->getEntityList()) {
        if (entity->hasComponent<Renderer>()) {
            addEntity(entity);
        }
    }
}


void RenderPass::onFrameRender(RenderContext& renderContext)
{

}


void RenderPass::onGuiRender(Gui& gui)
{

}


void RenderPass::setShaderName(const std::string& name)
{
    mShaderName = name;
    SYRINX_ENSURE(!mShaderName.empty());
}


void RenderPass::setCamera(Entity *camera)
{
    SYRINX_EXPECT(camera);
    SYRINX_EXPECT(camera->hasComponent<Transform>());
    SYRINX_EXPECT(camera->hasComponent<Camera>());
    mCamera = camera;
    SYRINX_ENSURE(mCamera);
}


void RenderPass::addEntity(Entity *entity)
{
    SYRINX_EXPECT(entity);
    mEntityList.push_back(entity);
}


void RenderPass::addEntityList(const std::vector<Entity *>& entityList)
{
    for (auto entity : entityList) {
        addEntity(entity);
    }
}


void RenderPass::setRenderState(RenderState *renderState)
{
    mRenderState = renderState;
}


const std::string& RenderPass::getName() const
{
    return mName;
}


const std::string& RenderPass::getShaderName() const
{
    return mShaderName;
}


Entity* RenderPass::getCamera() const
{
    return mCamera;
}


const RenderPass::EntityList& RenderPass::getEntityList() const
{
    return mEntityList;
}


RenderState* RenderPass::getRenderState() const
{
    return mRenderState;
}


bool RenderPass::isValid() const
{
    return mRenderState && !mEntityList.empty() && mCamera;
}

} // namespace Syrinx