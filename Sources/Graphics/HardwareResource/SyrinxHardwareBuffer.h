#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include <better-enums/enum.h>
#include <Common/SyrinxAssert.h>
#include <Exception/SyrinxException.h>
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
    ~HardwareBuffer() override;

    bool create() override;
    void read(size_t offset, size_t size, void *destination);
    void write(size_t offset, size_t size, const void *source);
    void setSize(size_t sizeInBytes);
    void setUsage(BufferUsage usage);
    template <typename T> void initData(const T *sourceData);
    template <typename T> void initData(const T *elemArray, size_t numElements);
    template <typename T> void setData(const T *sourceData);
    template <typename T> void setData(const T *elemArray, size_t numElements);
    template <typename T> void setData(size_t offset, const T *elemArray, size_t numElements);
    void uploadToGpu(size_t offset, size_t size);
    void uploadToGpu();
    size_t getSize() const;
    BufferUsage getUsage() const;
    template <typename T = uint8_t> const T* getData() const;
    template <typename T = uint8_t> T* getData();
    bool isImmutable() const;

protected:
    bool isValidToCreate() const override;

private:
    size_t mSizeInBytes;
    BufferUsage mUsage;
    std::unique_ptr<uint8_t[]> mData;
};


template <typename T>
void HardwareBuffer::initData(const T *sourceData)
{
    SYRINX_EXPECT(!isCreated());
    SYRINX_EXPECT(!mData);
    SYRINX_EXPECT(mSizeInBytes > 0);
    mData = std::unique_ptr<uint8_t[]>(new uint8_t[mSizeInBytes]);
    if (!sourceData) {
        std::memset(mData.get(), 0, mSizeInBytes);
    } else {
        setData(sourceData);
    }
    SYRINX_ENSURE(mData);
}


template <typename T>
void HardwareBuffer::initData(const T *elemArray, size_t numElements)
{
    SYRINX_EXPECT(!isCreated());
    SYRINX_EXPECT(elemArray && !mData);
    SYRINX_EXPECT(mSizeInBytes > 0);
    mData = std::unique_ptr<uint8_t[]>(new uint8_t[mSizeInBytes]);
    setData(elemArray, numElements);
    SYRINX_ENSURE(mData);
}


template <typename T>
void HardwareBuffer::setData(const T *sourceData)
{
    SYRINX_EXPECT(sourceData && mData);
    SYRINX_EXPECT(mSizeInBytes > 0);
    auto *source = reinterpret_cast<const uint8_t*>(sourceData);
    std::copy(source, source + mSizeInBytes, mData.get());
}


template <typename T>
void HardwareBuffer::setData(const T *elemArray, size_t numElements)
{
    SYRINX_EXPECT(elemArray && mData);
    SYRINX_EXPECT(mSizeInBytes > 0 && numElements > 0);

    size_t elementArraySize = sizeof(T) * numElements;
    if (elementArraySize > mSizeInBytes) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
            "fail to set data for buffer [{}] because the size of data is greater than the buffer size", getName());
    }
    auto *source = reinterpret_cast<const uint8_t*>(elemArray);
    std::copy(source, source + elementArraySize, mData.get());
}


template <typename T>
void HardwareBuffer::setData(size_t offset, const T *elemArray, size_t numElements)
{
    SYRINX_EXPECT(elemArray && mData);
    SYRINX_EXPECT(mSizeInBytes > 0 && numElements > 0);
    size_t elementArraySize = sizeof(T) * numElements;
    if (offset + elementArraySize > mSizeInBytes) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                   "fail to set data for buffer [{}] because the size of data is greater than the buffer size", getName());
    }
    auto source = reinterpret_cast<const uint8_t*>(elemArray);
    std::copy(source, source + elementArraySize, mData.get() + offset);
}


template <typename T>
const T* HardwareBuffer::getData() const
{
    return reinterpret_cast<T*>(mData.get());
}


template <typename T>
T* HardwareBuffer::getData()
{
    return reinterpret_cast<T*>(mData.get());
}

} // namespace Syrinx
