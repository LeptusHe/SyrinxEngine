#include "HardwareResource/SyrinxHardwareIndexBuffer.h"
#include <Common/SyrinxAssert.h>
#include <Logging/SyrinxLogManager.h>

namespace Syrinx {

size_t HardwareIndexBuffer::getSizeInBytesForIndexType(IndexType indexType)
{
    return indexType._value;
}


HardwareIndexBuffer::HardwareIndexBuffer(std::unique_ptr<HardwareBuffer>&& hardwareBuffer)
    : mIndexType(IndexType::UINT32)
    , mNumIndexes(0)
    , mHardwareBuffer(std::move(hardwareBuffer))
{
    SYRINX_ENSURE(mIndexType._value == IndexType::UINT32);
    SYRINX_ENSURE(mNumIndexes == 0);
    SYRINX_ENSURE(mHardwareBuffer);
    SYRINX_ENSURE(mHardwareBuffer->isImmutable());
    SYRINX_ENSURE(mHardwareBuffer->getSize() == 0);
}


void HardwareIndexBuffer::create()
{
    SYRINX_EXPECT(mHardwareBuffer);
    mHardwareBuffer->create();
}


void HardwareIndexBuffer::setIndexType(IndexType type)
{
    mIndexType = type;
    auto bufferSize = mNumIndexes * getSizeInBytesForIndexType(mIndexType);
    mHardwareBuffer->setSize(bufferSize);
    SYRINX_ENSURE(mIndexType._value == type);
    SYRINX_ENSURE(mHardwareBuffer->getSize() == bufferSize);
}


void HardwareIndexBuffer::setIndexNumber(size_t numIndexes)
{
    SYRINX_EXPECT(numIndexes > 0);
    mNumIndexes = numIndexes;
    auto bufferSize = mNumIndexes * getSizeInBytesForIndexType(mIndexType);
    mHardwareBuffer->setSize(bufferSize);
    SYRINX_ENSURE(mNumIndexes == numIndexes);
    SYRINX_ENSURE(mHardwareBuffer->getSize() == bufferSize);
}


const HardwareBuffer& HardwareIndexBuffer::getBuffer() const
{
    SYRINX_EXPECT(mHardwareBuffer);
    return *mHardwareBuffer;
}


HardwareResource::ResourceHandle HardwareIndexBuffer::getHandle() const
{
    SYRINX_EXPECT(mHardwareBuffer);
    return mHardwareBuffer->getHandle();
}


IndexType HardwareIndexBuffer::getIndexType() const
{
    return mIndexType;
}


size_t HardwareIndexBuffer::getNumIndexes() const
{
    return mNumIndexes;
}


bool HardwareIndexBuffer::isCreated() const
{
    SYRINX_EXPECT(mHardwareBuffer);
    return mHardwareBuffer->isCreated();
}

} // namespace Syrinx