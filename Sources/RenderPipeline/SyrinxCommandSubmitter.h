#pragma once
#include "RenderPipeline/SyrinxRenderPipeline.h"

namespace Syrinx {

class CommandSubmitter {
public:
    CommandSubmitter();
    ~CommandSubmitter() = default;
    void submit(const RenderPipeline& renderPipeline);

private:
    void submitCommandsForRenderPass(const RenderPass& renderPass);
    void submitCommandsToSetRenderState(const RenderState& renderState);
    void submitCommandsToDrawEntity(const Entity& entity);
    void submitCommandsToBindShaderPass(const ShaderPass& shaderPass);
    void submitCommandsToSetMVPMatrix(const Entity& entity, const ShaderPass& shaderPass);
    void submitCommandsToUpdateShaderParameter(ProgramStage& programStage, const ShaderParameter& shaderParameter);
    void submitCommandsToDrawMesh(const Mesh& mesh);
    void resetToDefaultState();

private:
    const RenderPipeline *mRenderPipeline;
    const RenderPass *mRenderPass;
    const Material* mMaterial;
};

} // namespace Syrinx