#include "RenderPipeline/SyrinxRenderPass.h"

namespace Syrinx {

RenderPass::RenderPass(const std::string& name)
    : mName(name)
    , mShaderPassName()
    , mCamera(nullptr)
    , mEntityList()
    , mRenderTarget(nullptr)
    , state()
{
    SYRINX_ENSURE(!mName.empty());
    SYRINX_ENSURE(mShaderPassName.empty());
    SYRINX_ENSURE(!mCamera);
    SYRINX_ENSURE(mEntityList.empty());
    SYRINX_ENSURE(!mRenderTarget);
}


void RenderPass::setShaderPassName(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    mShaderPassName = name;
    SYRINX_ENSURE(mShaderPassName == name);
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


void RenderPass::addEntityList(const std::vector<Entity*>& entityList)
{
    for (auto entity : entityList) {
        addEntity(entity);
    }
}


void RenderPass::setRenderTarget(const RenderTarget *renderTarget)
{
    SYRINX_EXPECT(!mRenderTarget);
    mRenderTarget = renderTarget;
    SYRINX_ENSURE(mRenderTarget);
    SYRINX_ENSURE(mRenderTarget->isCreated());
}


const std::string& RenderPass::getName() const
{
    return mName;
}


const std::string& RenderPass::getShaderPassName() const
{
    return mShaderPassName;
}


const RenderTarget* RenderPass::getRenderTarget() const
{
    SYRINX_EXPECT(isValid());
    return mRenderTarget;
}


Entity* RenderPass::getCamera() const
{
    return mCamera;
}


const RenderPass::EntityList& RenderPass::getEntityList() const
{
    return mEntityList;
}


bool RenderPass::isValid() const
{
    return mRenderTarget && !mEntityList.empty() && mCamera;
}

} // namespace Syrinx