#include "HardwareResource/SyrinxHardwareVertexBuffer.h"
#include <Common/SyrinxAssert.h>

namespace Syrinx {

HardwareVertexBuffer::HardwareVertexBuffer(std::unique_ptr<HardwareBuffer>&& hardwareBuffer)
    : mNumVertices(0)
    , mVertexSizeInBytes(0)
    , mHardwareBuffer(std::move(hardwareBuffer))
{
    SYRINX_ENSURE(mHardwareBuffer);
    SYRINX_ENSURE(mHardwareBuffer->isImmutable());
    SYRINX_ENSURE(mHardwareBuffer->getSize() == 0);
    SYRINX_ENSURE(mNumVertices == 0);
    SYRINX_ENSURE(mVertexSizeInBytes == 0);
}


void HardwareVertexBuffer::create()
{
    SYRINX_EXPECT(mHardwareBuffer);
    mHardwareBuffer->create();
}


void HardwareVertexBuffer::setVertexNumber(size_t numVertices)
{
    SYRINX_EXPECT(numVertices > 0);
    mNumVertices = numVertices;
    mHardwareBuffer->setSize(mNumVertices * mVertexSizeInBytes);
    SYRINX_ENSURE(mNumVertices == numVertices);
    SYRINX_ENSURE(mHardwareBuffer->getSize() == mNumVertices * mVertexSizeInBytes);
}


void HardwareVertexBuffer::setVertexSizeInBytes(size_t vertexSizeInBytes)
{
    SYRINX_EXPECT(vertexSizeInBytes > 0);
    mVertexSizeInBytes = vertexSizeInBytes;
    mHardwareBuffer->setSize(mNumVertices * mVertexSizeInBytes);
    SYRINX_ENSURE(mVertexSizeInBytes == vertexSizeInBytes);
    SYRINX_ENSURE(mHardwareBuffer->getSize() == mNumVertices * mVertexSizeInBytes);
}


void HardwareVertexBuffer::uploadToGpu(size_t offset, size_t size)
{
    mHardwareBuffer->uploadToGpu(offset, size);
}


void HardwareVertexBuffer::uploadToGpu()
{
    mHardwareBuffer->uploadToGpu();
}


size_t HardwareVertexBuffer::getSize() const
{
    SYRINX_EXPECT(isCreated());
    return mHardwareBuffer->getSize();
}


size_t HardwareVertexBuffer::getVertexNumber() const
{
    return mNumVertices;
}


size_t HardwareVertexBuffer::getVertexSizeInBytes() const
{
    return mVertexSizeInBytes;
}


HardwareResource::ResourceHandle HardwareVertexBuffer::getHandle() const
{
    SYRINX_EXPECT(mHardwareBuffer);
    return mHardwareBuffer->getHandle();
}


const HardwareBuffer& HardwareVertexBuffer::getBuffer() const
{
    SYRINX_EXPECT(mHardwareBuffer);
    return *mHardwareBuffer;
}


bool HardwareVertexBuffer::isCreated() const
{
    SYRINX_EXPECT(mHardwareBuffer);
    return mHardwareBuffer->isCreated();
}

} // namespace Syrinx