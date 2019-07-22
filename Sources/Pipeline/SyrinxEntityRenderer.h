#pragma once
#include <Graphics/SyrinxRenderContext.h>
#include <Scene/SyrinxEntity.h>
#include <Scene/Component/SyrinxRenderer.h>
#include <Scene/Component/SyrinxCamera.h>

namespace Syrinx {

class EntityRenderer {
public:
    void render(const Camera& camera, RenderContext& renderContext, Entity& renderer, const std::string& shaderName);

private:
    void setMatrices(const Camera& camera, Entity& entity, ShaderVars& shader);
    void updateParameters(ShaderVars& shaderVars);
    void drawMesh(const Mesh& mesh, RenderContext& renderContext);
};

} // namespace Syrinx