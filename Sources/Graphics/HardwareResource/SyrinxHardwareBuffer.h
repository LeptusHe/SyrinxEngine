#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include <better-enums/enum.h>
#include <Common/SyrinxAssert.h>
#include "HardwareResource/SyrinxHardwareResource.h"

namespace Syrinx {

BETTER_ENUM(BufferUsage, std::uint8_t,
    STATIC_DRAW,
    STATIC_READ,
    STATIC_COPY,
    DYNAMIC_DRAW,
    DYNAMIC_READ,
    DYNAMIC_COPY,
    DYNAMIC_IMMUTABLE
);


class HardwareBuffer : public HardwareResource {
public:
    explicit HardwareBuffer(const std::string& name);
    ~HardwareBuffer() override = default;

    bool create() override;
    void read(size_t offset, size_t size, void *destination);
    void write(size_t offset, size_t size, const void *source);
    void setSize(size_t sizeInBytes);
    void setUsage(BufferUsage usage);
    template <typename T> void setData(const T *sourceData);
    template <typename T> void setData(const T *elemArray, size_t numElements);
    size_t getSize() const;
    BufferUsage getUsage() const;
    template<typename T = uint8_t> const T* getData() const;
    bool isImmutable() const;

protected:
    bool isValidToCreate() const override;

private:
    size_t mSizeInBytes;
    BufferUsage mUsage;
    std::unique_ptr<uint8_t[]> mData;
};


template <typename T>
void HardwareBuffer::setData(const T *sourceData)
{
    SYRINX_EXPECT(!isCreated());
    SYRINX_EXPECT(sourceData && !mData);
    SYRINX_EXPECT(mSizeInBytes > 0);
    mData = std::unique_ptr<uint8_t[]>(new uint8_t[mSizeInBytes]);
    auto *source = reinterpret_cast<const uint8_t*>(sourceData);
    std::copy(source, source + mSizeInBytes, mData.get());
    SYRINX_ENSURE(mData);
}


template <typename T>
void HardwareBuffer::setData(const T *elemArray, size_t numElements)
{
    SYRINX_EXPECT(elemArray && !mData);
    SYRINX_EXPECT(!isCreated());
    SYRINX_EXPECT(mSizeInBytes > 0);
    mData = std::unique_ptr<uint8_t[]>(new uint8_t[mSizeInBytes]);
    size_t elementArraySize = sizeof(T) * numElements;
    auto *source = reinterpret_cast<const uint8_t*>(elemArray);
    std::copy(source, source + elementArraySize, mData.get());
    SYRINX_ENSURE(mData);
}


template <typename T>
const T* HardwareBuffer::getData() const
{
    return reinterpret_cast<T*>(mData.get());
}

} // namespace Syrinx
