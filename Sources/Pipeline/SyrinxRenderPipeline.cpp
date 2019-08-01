#include "SyrinxRenderPipeline.h"

namespace Syrinx {

RenderPipeline::RenderPipeline(const std::string& name)
    : IScriptableRenderPipeline(name)
    , mRenderPassList()
{
    SYRINX_ENSURE(mRenderPassList.empty());
}


void RenderPipeline::onInit(Scene *scene)
{
    for (auto pass : mRenderPassList) {
        pass->onInit(scene);
    }
}


void RenderPipeline::onFrameRender(RenderContext& renderContext)
{
    for (auto pass : mRenderPassList) {
        pass->onFrameRender(renderContext);
    }
}


void RenderPipeline::onGuiRender(Gui& gui)
{
    for (auto pass : mRenderPassList) {
        pass->onGuiRender(gui);
    }
}


void RenderPipeline::addRenderPass(RenderPass *renderPass)
{
    SYRINX_EXPECT(renderPass);
    mRenderPassList.push_back(renderPass);
}


const RenderPipeline::RenderPassList& RenderPipeline::getRenderPassList() const
{
    return mRenderPassList;
}

} // namespace Syrinx