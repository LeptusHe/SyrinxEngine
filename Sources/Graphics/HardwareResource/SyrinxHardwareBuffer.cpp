#include "HardwareResource/SyrinxHardwareBuffer.h"
#include <Common/SyrinxAssert.h>
#include <Logging/SyrinxLogManager.h>

namespace Syrinx {

HardwareBuffer::HardwareBuffer(const std::string& name)
    : HardwareResource(name)
    , mSizeInBytes(0)
    , mUsage(BufferUsage::DYNAMIC_IMMUTABLE)
    , mData(nullptr)
{
    SYRINX_ENSURE(isImmutable());
    SYRINX_ENSURE(mSizeInBytes == 0);
    SYRINX_ENSURE(!mData);
}


HardwareBuffer::~HardwareBuffer()
{
    auto handle = getHandle();
    glDeleteBuffers(1, &handle);
}


bool HardwareBuffer::create()
{
    SYRINX_EXPECT(!isCreated());
    SYRINX_EXPECT(isValidToCreate());

    GLuint handle = 0;
    glCreateBuffers(1, &handle);
    glNamedBufferStorage(handle, mSizeInBytes, mData.get(), GL_DYNAMIC_STORAGE_BIT);
    setHandle(handle);

    SYRINX_ENSURE(isCreated());
    return true;
}


void HardwareBuffer::read(size_t offset, size_t size, void *destination)
{
    SYRINX_EXPECT(isCreated());
    SYRINX_EXPECT(offset + size < mSizeInBytes);
    SYRINX_EXPECT(destination);

    SYRINX_ASSERT(false && "unimplemented");
}


void HardwareBuffer::write(size_t offset, size_t size, const void *source)
{
    SYRINX_EXPECT(isCreated());
    SYRINX_EXPECT(offset + size < mSizeInBytes);
    SYRINX_EXPECT(source);

    SYRINX_ASSERT(false && "unimplemented");
}


void HardwareBuffer::setSize(size_t sizeInBytes)
{
    SYRINX_EXPECT(!isCreated());
    mSizeInBytes = sizeInBytes;
    SYRINX_ENSURE(mSizeInBytes == sizeInBytes);
}


void HardwareBuffer::setUsage(BufferUsage usage)
{
    SYRINX_EXPECT(!isCreated());
    mUsage = usage;
    SYRINX_ENSURE(mUsage._value == usage);
}


size_t HardwareBuffer::getSize() const
{
    return mSizeInBytes;
}


BufferUsage HardwareBuffer::getUsage() const
{
    return mUsage;
}


bool HardwareBuffer::isImmutable() const
{
    return mUsage._value == BufferUsage::DYNAMIC_IMMUTABLE;
}


bool HardwareBuffer::isValidToCreate() const
{
    return mSizeInBytes > 0;
}

} // namespace Syrinx
