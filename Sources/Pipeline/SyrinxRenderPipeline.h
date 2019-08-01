#pragma once
#include <vector>
#include <GUI/SyrinxGui.h>
#include "SyrinxRenderPass.h"
#include "SyrinxScriptableRenderPipeline.h"

namespace Syrinx {

class RenderPipeline : public IScriptableRenderPipeline {
public:
    using RenderPassList = std::vector<RenderPass*>;

public:
    explicit RenderPipeline(const std::string& name);
    ~RenderPipeline() = default;

    void onInit(Scene *scene) override;
    void onFrameRender(RenderContext& renderContext) override;
    void onGuiRender(Gui& gui) override;
    void addRenderPass(RenderPass *renderPass);
    const RenderPassList& getRenderPassList() const;

private:
    RenderPassList mRenderPassList;
};

} // namespace Syrinx