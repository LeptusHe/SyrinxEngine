#include "SyrinxRenderPipeline.h"

namespace Syrinx {

RenderPipeline::RenderPipeline(const std::string& name)
    : mName(name)
    , mRenderPassList()
{
    SYRINX_ENSURE(!mName.empty());
    SYRINX_ENSURE(mRenderPassList.empty());
}


void RenderPipeline::addRenderPass(RenderPass *renderPass)
{
    SYRINX_EXPECT(renderPass);
    mRenderPassList.push_back(renderPass);
}


const std::string& RenderPipeline::getName() const
{
    return mName;
}

const RenderPipeline::RenderPassList& RenderPipeline::getRenderPassList() const
{
    return mRenderPassList;
}

} // namespace Syrinx