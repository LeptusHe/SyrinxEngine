#pragma once
#include <vector>
#include "SyrinxRenderPass.h"

namespace Syrinx {

class RenderPipeline {
public:
    using RenderPassList = std::vector<RenderPass*>;

public:
    explicit RenderPipeline(const std::string& name);
    ~RenderPipeline() = default;

    void addRenderPass(RenderPass *renderPass);
    const std::string& getName() const;
    const RenderPassList& getRenderPassList() const;

private:
    std::string mName;
    RenderPassList mRenderPassList;
};

} // namespace Syrinx