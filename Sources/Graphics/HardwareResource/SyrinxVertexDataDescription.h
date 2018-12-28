#pragma once
#include "HardwareResource/SyrinxHardwareVertexBuffer.h"

namespace Syrinx {

using VertexBufferBindingPoint = uint8_t;

class VertexDataDescription {
public:
    VertexDataDescription(const HardwareVertexBuffer *vertexBuffer,
                          VertexBufferBindingPoint bindingPoint,
                          size_t offsetOfFirstElement,
                          size_t strideBetweenElement);
    ~VertexDataDescription() = default;

    const HardwareVertexBuffer* getVertexBuffer() const;
    VertexBufferBindingPoint getVertexBufferBindingPoint() const;
    size_t getOffsetOfFirstElement() const;
    size_t getStrideBetweenElements() const;

private:
    const HardwareVertexBuffer *mVertexBuffer;
    VertexBufferBindingPoint mBindingPoint;
    size_t mOffsetOfFirstElement;
    size_t mStrideBetweenElement;
};

} // namespace Syrinx
