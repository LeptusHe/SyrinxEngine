#pragma once
#include <vector>
#include <HardwareResource/SyrinxVertexInputState.h>
#include <HardwareResource/SyrinxProgramPipeline.h>
#include <RenderResource/SyrinxMesh.h>
#include <RenderResource/SyrinxModel.h>
#include <Scene/SyrinxEntity.h>
#include <Scene/Component/SyrinxCamera.h>
#include "RenderPipeline/SyrinxRenderState.h"
#include "RenderResource/SyrinxRenderTarget.h"

namespace Syrinx {

class RenderPass {
public:
    using EntityList = std::vector<Entity*>;

public:
    explicit RenderPass(const std::string& name);
    ~RenderPass() = default;

    void setShaderPassName(const std::string& name);
    void setCamera(Camera *camera);
    void addEntity(Entity* entity);
    void addEntityList(const std::vector<Entity*>& entityList);
    void setRenderTarget(const RenderTarget *renderTarget);
    const std::string& getName() const;
    const std::string& getShaderPassName() const;
    const RenderTarget* getRenderTarget() const;
    Camera* getCamera() const;
    const EntityList& getEntityList() const;
    bool isValid() const;

private:
    std::string mName;
    std::string mShaderPassName;
    Camera *mCamera;
    EntityList mEntityList;
    const RenderTarget *mRenderTarget;

public:
    RenderState state;
};

} // namespace Syrinx