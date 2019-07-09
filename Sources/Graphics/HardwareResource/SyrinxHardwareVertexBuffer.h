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
    template <typename T> void initData(const T *sourceData);
    template <typename T> void initData(const T *elemArray, size_t numElements);
    template <typename T> void setData(const T *data);
    template <typename T> void setData(const T *elemArray, size_t numElements);
    template <typename T> void setData(size_t offset, const T* elemArray, size_t numElements);
    void uploadToGpu(size_t offset, size_t size);
    void uploadToGpu();
    size_t getSize() const;
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
void HardwareVertexBuffer::initData(const T *sourceData)
{
    mHardwareBuffer->initData(sourceData);
}


template <typename T>
void HardwareVertexBuffer::initData(const T *elemArray, size_t numElements)
{
    mHardwareBuffer->initData(elemArray, numElements);
}


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


template <typename T>
void HardwareVertexBuffer::setData(size_t offset, const T *elemArray, size_t numElements)
{
    mHardwareBuffer->setData(offset, elemArray, numElements);
}

} // namespace Syrinx
