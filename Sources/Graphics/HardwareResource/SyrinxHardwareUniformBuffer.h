#pragma once
#include <memory>
#include "HardwareResource/SyrinxHardwareBuffer.h"

namespace Syrinx {

class HardwareUniformBuffer {
public:
    explicit HardwareUniformBuffer(std::unique_ptr<HardwareBuffer>&& hardwareBuffer);
    ~HardwareUniformBuffer() = default;

    void setSize(size_t size);
    size_t getSize() const;
    void create();
    template <typename T> void initData(const T *sourceData);
    template <typename T> void initData(const T *elemArray, size_t numElements);
    template <typename T> void setData(size_t offset, const T& data);
    void setData(size_t offset, const uint8_t* data, size_t sizeInBytes);
    const HardwareBuffer& getBuffer() const;
    uint8_t *getData();
    HardwareResource::ResourceHandle getHandle() const;
    bool isCreated() const;
    void uploadToGpu();

private:
    std::unique_ptr<HardwareBuffer> mHardwareBuffer;
};


template <typename T>
void HardwareUniformBuffer::initData(const T *sourceData)
{
    mHardwareBuffer->initData(sourceData);
}


template <typename T>
void HardwareUniformBuffer::initData(const T *elemArray, size_t numElements)
{
    mHardwareBuffer->initData(elemArray, numElements);
}


template <typename T>
void HardwareUniformBuffer::setData(size_t offset, const T& data)
{
    auto sizeInBytes = sizeof(T);
    setData(offset, reinterpret_cast<const uint8_t*>(&data), sizeInBytes);
}

} // namespace Syrinx