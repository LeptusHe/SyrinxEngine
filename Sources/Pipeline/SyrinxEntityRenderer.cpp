#include "SyrinxEntityRenderer.h"

namespace Syrinx {

void EntityRenderer::render(const Camera& camera, RenderContext& renderContext, Entity& entity, const std::string& shaderName)
{
    auto renderState = renderContext.getRenderState();
    if (!renderState) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
            "fail to render entity [{}] because there is no render state in render context", entity.getName());
    }

    if (!entity.hasComponent<Renderer>()) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
            "fail to render entity [{}] because it doesn't have renderer component", entity.getName());
    }

    auto& renderer = entity.getComponent<Renderer>();
    const auto& mesh = renderer.getMesh();
    auto material = renderer.getMaterial();

    ShaderVars* shaderVars = material->getShaderVars(shaderName);
    if (!shaderVars) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                   "fail to render mesh [{}] because it doesn't have shader [{}] in material [{}]",
                                   renderer.getMesh()->getName(), shaderName, material->getName());
    }

    auto programPipeline = shaderVars->getShader().getProgramPipeline();
    renderState->setProgramPipeline(programPipeline);
    renderContext.prepareDraw();

    setMatrices(camera, entity, *shaderVars);
    updateParameters(*shaderVars);
    drawMesh(*renderer.getMesh(), renderContext);
}


void EntityRenderer::setMatrices(const Camera& camera, Entity& entity, ShaderVars& shaderVars)
{
    auto projectionMat = camera.getProjectionMatrix();
    auto viewMat = camera.getViewMatrix();

    if (!entity.hasComponent<Transform>()) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                   "fail to render mesh [{}] because it doesn't have transform component", entity.getName());
    }

    const auto& transform = entity.getComponent<Transform>();
    auto worldMatrix = transform.getWorldMatrix();

    auto& shader = shaderVars.getShader();
    auto vertexStage = shader.getShaderModule(ProgramStageType::VertexStage);
    auto& vertexStageVars = *(shaderVars.getProgramVars(ProgramStageType::VertexStage));
    vertexStageVars["SyrinxMatrixBuffer"]["SYRINX_MATRIX_PROJ"] = projectionMat;
    vertexStageVars["SyrinxMatrixBuffer"]["SYRINX_MATRIX_VIEW"] = viewMat;
    vertexStageVars["SyrinxMatrixBuffer"]["SYRINX_MATRIX_WORLD"] = worldMatrix;

    vertexStage->updateProgramVars(vertexStageVars);
    vertexStage->uploadParametersToGpu();
    vertexStage->bindResources();
}


void EntityRenderer::updateParameters(ShaderVars& shaderVars)
{
    auto shader = shaderVars.getShader();
    auto fragmentStage = shader.getShaderModule(ProgramStageType::FragmentStage);
    auto fragmentVars = shaderVars.getProgramVars(ProgramStageType::FragmentStage);

    fragmentStage->updateProgramVars(*fragmentVars);
    fragmentStage->uploadParametersToGpu();
    fragmentStage->bindResources();
}


void EntityRenderer::drawMesh(const Mesh& mesh, RenderContext& renderContext)
{
    const auto& vertexInputState = mesh.getVertexInputState();
    auto renderState = renderContext.getRenderState();
    renderState->setVertexInputState(&vertexInputState);
    renderContext.drawIndexed(3 * mesh.getNumTriangle());
}

} // namespace Syrinx
