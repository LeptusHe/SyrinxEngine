#include "HardwareResource/SyrinxVertexDataDescription.h"
#include <Common/SyrinxAssert.h>

namespace Syrinx {

VertexDataDescription::VertexDataDescription(const HardwareVertexBuffer *vertexBuffer,
                                             VertexBufferBindingPoint bindingPoint,
                                             size_t offsetOfFirstElement,
                                             size_t strideBetweenElement)
    : mVertexBuffer(vertexBuffer)
    , mBindingPoint(bindingPoint)
    , mOffsetOfFirstElement(offsetOfFirstElement)
    , mStrideBetweenElement(strideBetweenElement)
{
    SYRINX_ENSURE(mVertexBuffer);
    SYRINX_ENSURE(mVertexBuffer->isCreated());
    SYRINX_ENSURE(mOffsetOfFirstElement >= 0);
    SYRINX_ENSURE(mStrideBetweenElement > 0);
}


const HardwareVertexBuffer* VertexDataDescription::getVertexBuffer() const
{
    SYRINX_EXPECT(mVertexBuffer);
    return mVertexBuffer;
}


VertexBufferBindingPoint VertexDataDescription::getVertexBufferBindingPoint() const
{
    SYRINX_EXPECT(mVertexBuffer);
    return mBindingPoint;
}


size_t VertexDataDescription::getOffsetOfFirstElement() const
{
    SYRINX_EXPECT(mVertexBuffer);
    return mOffsetOfFirstElement;
}


size_t VertexDataDescription::getStrideBetweenElements() const
{
    SYRINX_EXPECT(mVertexBuffer && mStrideBetweenElement > 0);
    return mStrideBetweenElement;
}

} // namespace Syrinx
