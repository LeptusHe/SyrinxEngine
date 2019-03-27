#include "RenderResource/SyrinxResource.h"
#include <Common/SyrinxAssert.h>

namespace Syrinx {

Resource::Resource(const std::string& name) : mName(name)
{
    SYRINX_ENSURE(!mName.empty());
}


const std::string& Resource::getName() const
{
    return mName;
}

} // namespace Syrinx