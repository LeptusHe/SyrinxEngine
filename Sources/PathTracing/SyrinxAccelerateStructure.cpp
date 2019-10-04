#include "SyrinxAccelerateStructure.h"
#include <Common/SyrinxAssert.h>

namespace Syrinx {

AccelerateStructure::AccelerateStructure(AccelerateStructure&& rhs) noexcept
    : mHandle(rhs.mHandle)
    , mBuffer(std::move(rhs.mBuffer))
{
    rhs.mHandle = 0;
    SYRINX_ENSURE(rhs.mBuffer.getDevicePtr() == 0);
}

} // namespace Syrinx