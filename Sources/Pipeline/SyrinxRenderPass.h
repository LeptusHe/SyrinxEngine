#pragma once
#include <vector>
#include <Graphics/SyrinxRenderState.h>
#include <Graphics/SyrinxRenderContext.h>
#include <Scene/SyrinxScene.h>
#include <Graphics/SyrinxRenderContext.h>
#include <Scene/SyrinxEntity.h>
#include <Scene/Component/SyrinxCamera.h>

namespace Syrinx {

class RenderPass {
public:
    using EntityList = std::vector<Entity*>;

public:
    explicit RenderPass(const std::string& name);
    ~RenderPass() = default;

    virtual void onInit(Scene& scene);
    virtual void onFrameRender(RenderContext& renderContext) = 0;
    void setShaderName(const std::string& name);
    void setCamera(Entity *camera);
    void addEntity(Entity *entity);
    void addEntityList(const std::vector<Entity*>& entityList);
    void setRenderState(RenderState *renderState);
    const std::string& getName() const;
    const std::string& getShaderName() const;
    Entity* getCamera() const;
    const EntityList& getEntityList() const;
    RenderState* getRenderState() const;
    bool isValid() const;

private:
    std::string mName;
    std::string mShaderName;
    Entity *mCamera;
    EntityList mEntityList;
    RenderState *mRenderState;
};

} // namespace Syrinx