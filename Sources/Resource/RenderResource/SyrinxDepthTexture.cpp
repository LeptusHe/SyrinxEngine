#include "RenderResource/SyrinxDepthTexture.h"
#include <Common/SyrinxAssert.h>

namespace Syrinx {

DepthTexture::DepthTexture(const std::string& name, const RenderTexture *depthTexture)
    : mName(name)
    , mDepthTexture(depthTexture)
{
    SYRINX_ENSURE(!mName.empty());
    SYRINX_ENSURE(mDepthTexture);
    SYRINX_ENSURE(mDepthTexture->isDepthTexture());
}


HardwareResource::ResourceHandle DepthTexture::getHandle() const
{
    return mDepthTexture->getHandle();
}


const std::string& DepthTexture::getName() const
{
    return mName;
}

} // namespace Syrinx