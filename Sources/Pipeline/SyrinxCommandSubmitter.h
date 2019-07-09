#pragma once
#include <Graphics/SyrinxRenderContext.h>
#include <Resource/RenderResource/SyrinxMesh.h>
#include <Resource/RenderResource/SyrinxShaderVars.h>
#include <Resource/ResourceManager/SyrinxShaderManager.h>
#include "SyrinxRenderPipeline.h"

namespace Syrinx {

class CommandSubmitter {
public:
    explicit CommandSubmitter(ShaderManager *shaderManager);
    ~CommandSubmitter() = default;
    void submit(const RenderPipeline& renderPipeline);

private:
    void submitCommandsForRenderPass(const RenderPass& renderPass);
    void submitCommandsToDrawEntity(Entity& entity);
    void submitCommandsToSetMatrices(Entity& entity, ShaderVars *shaderVars);
    void submitCommandsToUpdateParameters(ShaderVars *shaderVars);
    void submitCommandsToDrawMesh(const Mesh& mesh);
    void reset();

private:
    ShaderManager *mShaderManager;
    RenderContext mRenderContext;
    const RenderPipeline *mRenderPipeline;
    const RenderPass *mRenderPass;
};

} // namespace Syrinx