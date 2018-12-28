#pragma once
#include <vector>
#include <memory>
#include "HardwareResource/SyrinxHardwareBuffer.h"

namespace Syrinx {

class HardwareVertexBuffer {
public:
    explicit HardwareVertexBuffer(std::unique_ptr<HardwareBuffer>&& hardwareBuffer);
    ~HardwareVertexBuffer() = default;

    void create();
    void setVertexNumber(size_t numVertices);
    void setVertexSizeInBytes(size_t vertexSizeInBytes);
    template <typename T> void setData(const T *data);
    template <typename T> void setData(const T *elemArray, size_t numElements);
    size_t getVertexNumber() const;
    size_t getVertexSizeInBytes() const;
    HardwareResource::ResourceHandle getHandle() const;
    const HardwareBuffer& getBuffer() const;
    bool isCreated() const;

private:
    size_t mNumVertices;
    size_t mVertexSizeInBytes;
    std::unique_ptr<HardwareBuffer> mHardwareBuffer;
};


template <typename T>
void HardwareVertexBuffer::setData(const T *data)
{
    mHardwareBuffer->setData(data);
}


template <typename T>
void HardwareVertexBuffer::setData(const T *elemArray, size_t numElements)
{
    mHardwareBuffer->setData(elemArray, numElements);
}

} // namespace Syrinx
