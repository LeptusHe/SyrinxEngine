#include "HardwareResource/SyrinxHardwareUniformBuffer.h"
#include <Exception/SyrinxException.h>

namespace Syrinx {

HardwareUniformBuffer::HardwareUniformBuffer(std::unique_ptr<HardwareBuffer>&& hardwareBuffer)
    : mHardwareBuffer(std::move(hardwareBuffer))
{
    SYRINX_ENSURE(mHardwareBuffer);
    SYRINX_ENSURE(mHardwareBuffer->isImmutable());
    SYRINX_ENSURE(mHardwareBuffer->getSize() == 0);
}


void HardwareUniformBuffer::setSize(size_t size)
{
    mHardwareBuffer->setSize(size);
}


size_t HardwareUniformBuffer::getSize() const
{
    return mHardwareBuffer->getSize();
}


void HardwareUniformBuffer::create()
{
    SYRINX_EXPECT(mHardwareBuffer);
    mHardwareBuffer->create();
}


const HardwareBuffer& HardwareUniformBuffer::getBuffer() const
{
    SYRINX_EXPECT(mHardwareBuffer);
    return *mHardwareBuffer;
}


uint8_t* HardwareUniformBuffer::getData()
{
    SYRINX_EXPECT(mHardwareBuffer);
    SYRINX_EXPECT(isCreated());
    return mHardwareBuffer->getData();
}


HardwareResource::ResourceHandle HardwareUniformBuffer::getHandle() const
{
    SYRINX_EXPECT(mHardwareBuffer);
    return mHardwareBuffer->getHandle();
}


bool HardwareUniformBuffer::isCreated() const
{
    SYRINX_EXPECT(mHardwareBuffer);
    return mHardwareBuffer->isCreated();
}


void HardwareUniformBuffer::uploadToGpu()
{
    SYRINX_EXPECT(isCreated());
    auto bufferHandle = mHardwareBuffer->getHandle();
    auto data = mHardwareBuffer->getData();
    glNamedBufferSubData(bufferHandle, 0, getSize(), data);
}


void HardwareUniformBuffer::setData(size_t offset, const uint8_t *data, size_t sizeInBytes)
{
    SYRINX_EXPECT(data);
    SYRINX_EXPECT(mHardwareBuffer);
    mHardwareBuffer->setData(offset, data, sizeInBytes);
}

} // namespace Syrinx