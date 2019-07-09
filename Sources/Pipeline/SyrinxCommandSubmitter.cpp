#include "SyrinxCommandSubmitter.h"
#include <Component/SyrinxRenderer.h>
#include <Component/SyrinxTransform.h>
#include <Exception/SyrinxException.h>

namespace Syrinx {

CommandSubmitter::CommandSubmitter(ShaderManager *shaderManager)
    : mShaderManager(shaderManager)
    , mRenderContext()
    , mRenderPipeline(nullptr)
    , mRenderPass(nullptr)
{
    SYRINX_ENSURE(mShaderManager);
    SYRINX_ENSURE(!mRenderPipeline);
    SYRINX_ENSURE(!mRenderPass);
}


void CommandSubmitter::submit(const RenderPipeline& renderPipeline)
{
    mRenderPipeline = &renderPipeline;
    for (const auto renderPass : renderPipeline.getRenderPassList()) {
        mRenderPass = renderPass;
        SYRINX_ASSERT(mRenderPass == renderPass);
        submitCommandsForRenderPass(*renderPass);
    }
    reset();
    SYRINX_ENSURE(!mRenderPipeline);
    SYRINX_ENSURE(!mRenderPass);
}


void CommandSubmitter::submitCommandsForRenderPass(const RenderPass& renderPass)
{
    auto shader = mShaderManager->find(renderPass.getShaderName());
    if (!shader) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
            "fail to execute render pass [{}] because shader [{}] doesn't exist", renderPass.getName(), renderPass.getShaderName());
    }

    RenderState *renderState = renderPass.getRenderState();
    renderState->setProgramPipeline(shader->getProgramPipeline());

    mRenderContext.setRenderState(renderState);
    mRenderContext.prepareDraw();

    for (auto entity : renderPass.getEntityList()) {
        submitCommandsToDrawEntity(*entity);
    }
}


void CommandSubmitter::submitCommandsToDrawEntity(Entity& entity)
{
    if (!entity.hasComponent<Renderer>()) {
        return;
    }

    auto& renderer = entity.getComponent<Renderer>();
    const Mesh *mesh = renderer.getMesh();

    auto material = renderer.getMaterial();
    auto shaderVars = material->getShaderVars(mRenderPass->getShaderName());
    if (!shaderVars) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
            "fail to render entity [{}] in render pass [{}] because it doesn't have shader [{}] in material [{}]",
            entity.getName(), mRenderPass->getName(), mRenderPass->getShaderName(), material->getName());
    }

    submitCommandsToSetMatrices(entity, shaderVars);
    submitCommandsToUpdateParameters(shaderVars);
    submitCommandsToDrawMesh(*mesh);
}


void CommandSubmitter::submitCommandsToSetMatrices(Entity& entity, ShaderVars *shaderVars)
{
    SYRINX_EXPECT(shaderVars);
    const auto& shader = shaderVars->getShader();
    auto cameraEntity = mRenderPass->getCamera();
    if (!mRenderPass->getCamera()) {
        return;
    }

    if (!cameraEntity->hasComponent<Camera>()) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
            "the camera entity [{}] of render pass [{}] doesn't have camera component", cameraEntity->getName(), mRenderPass->getName());
    }

    auto& camera = cameraEntity->getComponent<Camera>();
    auto projectionMat = camera.getProjectionMatrix();
    auto viewMat = camera.getViewMatrix();

    if (!entity.hasComponent<Transform>()) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
            "fail to execute render pass [{}] because entity [{}] doesn't have transform component", mRenderPass->getName(), entity.getName());
    }
    auto& transform = entity.getComponent<Transform>();
    auto worldMatrix = transform.getWorldMatrix();

    auto vertexStage = shader.getShaderModule(ProgramStageType::VertexStage);
    auto& vertexStageVars = *(shaderVars->getProgramVars(ProgramStageType::VertexStage));
    vertexStageVars["SyrinxMatrixBuffer"]["SYRINX_MATRIX_PROJ"] = projectionMat;
    vertexStageVars["SyrinxMatrixBuffer"]["SYRINX_MATRIX_VIEW"] = viewMat;
    vertexStageVars["SyrinxMatrixBuffer"]["SYRINX_MATRIX_WORLD"] = worldMatrix;

    vertexStage->updateProgramVars(vertexStageVars);
    vertexStage->uploadParametersToGpu();
    vertexStage->bindResources();
}


void CommandSubmitter::submitCommandsToUpdateParameters(ShaderVars *shaderVars)
{
    SYRINX_EXPECT(shaderVars);
    auto shader = shaderVars->getShader();
    auto fragmentStage = shader.getShaderModule(ProgramStageType::FragmentStage);
    auto fragmentVars = shaderVars->getProgramVars(ProgramStageType::FragmentStage);

    fragmentStage->updateProgramVars(*fragmentVars);
    fragmentStage->uploadParametersToGpu();
    fragmentStage->bindResources();
}


void CommandSubmitter::submitCommandsToDrawMesh(const Mesh& mesh)
{
    const auto& vertexInputState = mesh.getVertexInputState();
    auto renderState = mRenderPass->getRenderState();
    renderState->setVertexInputState(&vertexInputState);

    mRenderContext.drawIndexed(3 * mesh.getNumTriangle());
}


void CommandSubmitter::reset()
{
    mRenderContext.setRenderState(nullptr);
    mRenderPipeline = nullptr;
    mRenderPass = nullptr;
}

} // namespace Syrinx