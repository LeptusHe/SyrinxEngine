#pragma once
#include <cstdint>
#include <memory>
#include "HardwareResource/SyrinxHardwareBuffer.h"

namespace Syrinx {

BETTER_ENUM(IndexType, std::uint8_t,
    UINT8  = sizeof(uint8_t),
    UINT16 = sizeof(uint16_t),
    UINT32 = sizeof(uint32_t)
);


class HardwareIndexBuffer {
public:
    static size_t getSizeInBytesForIndexType(IndexType indexType);

public:
    explicit HardwareIndexBuffer(std::unique_ptr<HardwareBuffer>&& hardwareBuffer);
    ~HardwareIndexBuffer() = default;

    void create();
    void setIndexType(IndexType type);
    void setIndexNumber(size_t numIndexes);
    template <typename T> void setData(const T* source);
    template <typename T> void setData(const T* elemArray, size_t numElements);
    const HardwareBuffer& getBuffer() const;
    HardwareResource::ResourceHandle getHandle() const;
    IndexType getIndexType() const;
    size_t getNumIndexes() const;
    bool isCreated() const;

private:
    IndexType mIndexType;
    size_t mNumIndexes;
    std::unique_ptr<HardwareBuffer> mHardwareBuffer;
};


template <typename T>
void HardwareIndexBuffer::setData(const T *source)
{
    mHardwareBuffer->setData(source);
}


template <typename T>
void HardwareIndexBuffer::setData(const T *elemArray, size_t numElements)
{
    mHardwareBuffer->setData(elemArray, numElements);
}

} // namespace Syrinx
