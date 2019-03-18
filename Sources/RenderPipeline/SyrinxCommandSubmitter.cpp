#include "RenderPipeline/SyrinxCommandSubmitter.h"
#include <Component/SyrinxRenderer.h>
#include <Component/SyrinxTransform.h>
#include <Exception/SyrinxException.h>

namespace Syrinx {

namespace {

template <typename T>
void updateParameter(ProgramStage& programStage, const ShaderParameter& shaderParameter)
{
    const auto& parameterName = shaderParameter.getName();
    const auto& parameterValue = shaderParameter.getValue();
    if (auto value = std::get_if<T>(&parameterValue); value) {
        programStage.updateParameter(parameterName, *value);
    }
}

} // namespace anonymous




CommandSubmitter::CommandSubmitter()
    : mRenderPipeline(nullptr)
    , mRenderPass(nullptr)
    , mMaterial(nullptr)
{
    SYRINX_ENSURE(!mRenderPipeline);
    SYRINX_ENSURE(!mRenderPass);
    SYRINX_ENSURE(!mMaterial);
}


void CommandSubmitter::submit(const RenderPipeline& renderPipeline)
{
    mRenderPipeline = &renderPipeline;
    for (const auto renderPass : renderPipeline.getRenderPassList()) {
        mRenderPass = renderPass;
        SYRINX_ASSERT(mRenderPass == renderPass);
        submitCommandsForRenderPass(*renderPass);
    }
    resetToDefaultState();
    SYRINX_ENSURE(!mRenderPipeline);
    SYRINX_ENSURE(!mRenderPass);
    SYRINX_ENSURE(!mMaterial);
}


void CommandSubmitter::submitCommandsForRenderPass(const RenderPass& renderPass)
{
    submitCommandsToSetRenderState(renderPass.state);
    for (const auto& entity : renderPass.getEntityList()) {
        submitCommandsToDrawEntity(*entity);
    }
}


void CommandSubmitter::submitCommandsToSetRenderState(const RenderState& renderState)
{
    float defaultValueForColorAttachment[] = {0.5, 0.0, 0.5, 1.0};
    glClearNamedFramebufferfv(0, GL_COLOR, 0, defaultValueForColorAttachment);
    float defaultDepthValue = 1.0;
    glClearNamedFramebufferfv(0, GL_DEPTH, 0, &defaultDepthValue);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
}


void CommandSubmitter::submitCommandsToDrawEntity(const Entity& entity)
{
    if (!entity.hasComponent<const Renderer>()) {
        return;
    }

    const auto& renderer = entity.getComponent<const Renderer>();
    const Mesh *mesh = renderer.getMesh();
    mMaterial = renderer.getMaterial();
    SYRINX_ASSERT(mMaterial == renderer.getMaterial());

    auto shader = mMaterial->getShader();
    auto shaderPass = shader->getShaderPass(mRenderPass->getShaderPassName());
    if (!shaderPass) {
        return;
    }
    submitCommandsToSetMVPMatrix(entity, *shaderPass);
    submitCommandsToBindShaderPass(*shaderPass);
    submitCommandsToDrawMesh(*mesh);
}


void CommandSubmitter::submitCommandsToSetMVPMatrix(const Entity& entity, const ShaderPass& shaderPass)
{
    if (!entity.hasComponent<Transform>()) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidState, "entity [{}] doesn't have transform component", entity.getName());
    }
    if (!mRenderPass->getCamera()) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidState, "render pass [{}] doesn't have camera", mRenderPass->getName());
    }

    const auto& transform = entity.getComponent<const Transform>();
    auto vertexProgram = shaderPass.getProgramStage(ProgramStageType::VertexStage);

    Entity* cameraEntity = mRenderPass->getCamera();
    SYRINX_ASSERT(cameraEntity);
    const auto& cameraTransform = cameraEntity->getComponent<Transform>();
    const auto& cameraComponent = cameraEntity->getComponent<Camera>();

    Syrinx::Matrix4x4 viewMatrix(1.0f);
    viewMatrix = glm::translate(viewMatrix, -cameraTransform.getLocalPosition());
    viewMatrix = viewMatrix * cameraTransform.getRotateMatrix();

    vertexProgram->updateParameter("uModelMatrix", transform.getWorldMatrix());
    vertexProgram->updateParameter("uViewMatrix", viewMatrix);
    vertexProgram->updateParameter("uProjectionMatrix", cameraComponent.getProjectionMatrix());
}


void CommandSubmitter::submitCommandsToBindShaderPass(const ShaderPass& shaderPass)
{
    auto programPipeline = shaderPass.getProgramPipeline();
    glBindProgramPipeline(programPipeline->getHandle());
    for (const auto& [parameterName, ProgramStageList] : shaderPass.getParameterReferenceMap()) {
        ShaderParameter *shaderParameter = shaderPass.getParameter(parameterName);
        if (auto materialParameter = mMaterial->getMaterialParameter(shaderParameter->getName()); materialParameter) {
            shaderParameter = materialParameter;
        }
        SYRINX_ASSERT(shaderParameter);
        for (auto programStage : ProgramStageList) {
            submitCommandsToUpdateShaderParameter(*programStage, *shaderParameter);
        }
    }
}


void CommandSubmitter::submitCommandsToUpdateShaderParameter(ProgramStage& programStage, const ShaderParameter& shaderParameter)
{
    updateParameter<int>(programStage, shaderParameter);
    updateParameter<float>(programStage, shaderParameter);
    updateParameter<Color>(programStage, shaderParameter);

    const auto& parameterName = shaderParameter.getName();
    const auto& parameterValue = shaderParameter.getValue();
    if (shaderParameter.getType()._value == ShaderParameterType::TEXTURE_2D) {
        const auto& textureValue = std::get<TextureValue>(parameterValue);
        glBindTextureUnit(static_cast<unsigned int>(textureValue.textureUnit), textureValue.texture->getHandle());
        programStage.updateParameter(shaderParameter.getName(), static_cast<int>(textureValue.textureUnit));
    }
}


void CommandSubmitter::submitCommandsToDrawMesh(const Mesh& mesh)
{
    const auto& vertexInputState = mesh.getVertexInputState();
    glBindVertexArray(vertexInputState.getHandle());
    glDrawElements(GL_TRIANGLES, static_cast<int>(3 * mesh.getNumTriangle()), GL_UNSIGNED_INT, nullptr);
}


void CommandSubmitter::resetToDefaultState()
{
    mRenderPipeline = nullptr;
    mRenderPass = nullptr;
    mMaterial = nullptr;
}

} // namespace Syrinx