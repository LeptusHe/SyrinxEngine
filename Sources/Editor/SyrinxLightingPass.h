#pragma once
#include <Pipeline/SyrinxRenderPass.h>
#include <Pipeline/SyrinxEntityRenderer.h>

namespace Syrinx {

class LightingPass : public RenderPass {
public:
    explicit LightingPass(const std::string& name) : RenderPass(name) { }

    void onFrameRender(RenderContext& renderContext) override
    {
        renderContext.clearRenderTarget(nullptr, Color(1.0, 0.0, 0.0, 1.0));
        renderContext.clearDepth(nullptr, 1.0);

        EntityRenderer entityRenderer;
        auto cameraEntity = getCamera();
        if (!cameraEntity || (!cameraEntity->hasComponent<Camera>())) {
            return;
        }

        renderContext.pushRenderState();
        renderContext.setRenderState(getRenderState());
        auto& camera = cameraEntity->getComponent<Camera>();
        for (auto entity : getEntityList()) {
            entityRenderer.render(camera, renderContext, *entity, getShaderName());
        }
        renderContext.popRenderState();
    }
};

} // namespace Syrinx