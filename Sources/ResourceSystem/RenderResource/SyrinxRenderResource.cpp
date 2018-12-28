#include "RenderResource/SyrinxRenderResource.h"
#include <Common/SyrinxAssert.h>

namespace Syrinx {

RenderResource::RenderResource(const std::string& name) : mName(name)
{
    SYRINX_ENSURE(!mName.empty());
}


const std::string& RenderResource::getName() const
{
    return mName;
}

} // namespace Syrinx