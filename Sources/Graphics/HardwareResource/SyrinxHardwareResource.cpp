#include "HardwareResource/SyrinxHardwareResource.h"
#include <Common/SyrinxAssert.h>

namespace Syrinx {

void HardwareResourceHandle::setHandleValue(GLuint value)
{
    SYRINX_EXPECT(value != INVALID_RESOURCE_HANDLE_VALUE);
    mHandleValue = value;
    SYRINX_ENSURE(mHandleValue != INVALID_RESOURCE_HANDLE_VALUE);
}


GLuint HardwareResourceHandle::getHandleValue() const
{
    return mHandleValue;
}


GLuint& HardwareResourceHandle::fetchHandleValue()
{
    return mHandleValue;
}


HardwareResource::HardwareResource(const std::string& name)
    : mName(name)
    , mHandle()
    , mState(HardwareResourceState::Uncreated)
{
    SYRINX_ENSURE(!name.empty());
    SYRINX_ENSURE(mName == name);
    SYRINX_ENSURE(mHandle.getHandleValue() == HardwareResourceHandle::INVALID_RESOURCE_HANDLE_VALUE);
    SYRINX_ENSURE(mState._value == HardwareResourceState::Uncreated);
}


void HardwareResource::setHandle(HardwareResource::ResourceHandle handle)
{
    mHandle.setHandleValue(handle);
    setState(HardwareResourceState::Created);
}


const std::string& HardwareResource::getName() const
{
    return mName;
}


HardwareResourceState HardwareResource::getState() const
{
    return mState;
}


HardwareResource::ResourceHandle HardwareResource::getHandle() const
{
    return mHandle.getHandleValue();
}


HardwareResource::ResourceHandle& HardwareResource::fetchHandle()
{
    return mHandle.fetchHandleValue();
}


bool HardwareResource::isCreated() const
{
    return mState._value != HardwareResourceState::Uncreated;
}


void HardwareResource::setState(HardwareResourceState state)
{
    mState = state;
}

} // namespace Syrinx
