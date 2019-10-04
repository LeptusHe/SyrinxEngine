#include "SyrinxCudaBuffer.h"
#include <Common/SyrinxAssert.h>
#include "SyrinxCudaAssert.h"

namespace Syrinx {

CudaBuffer::CudaBuffer(CudaBuffer&& rhs) noexcept
    : Resource(rhs.getName())
{
    mSizeInBytes = rhs.mSizeInBytes;
    mDeviceMemory = rhs.mDeviceMemory;
    rhs.mSizeInBytes = 0;
    rhs.mDeviceMemory = nullptr;
}


CudaBuffer::~CudaBuffer()
{
    if (mDeviceMemory != nullptr) {
        free();
    }
}


void CudaBuffer::allocate(size_t sizeInBytes)
{
    SYRINX_ASSERT((mDeviceMemory == nullptr) && (sizeInBytes > 0) );
    SYRINX_CUDA_ASSERT(cudaMalloc((void**)(&mDeviceMemory), sizeInBytes));
    mSizeInBytes = sizeInBytes;
}


void CudaBuffer::resize(size_t sizeInBytes)
{
    SYRINX_ASSERT(sizeInBytes > 0);
    if (mDeviceMemory) {
        free();
    }
    allocate(sizeInBytes);
}


void CudaBuffer::free()
{
    SYRINX_ASSERT(mDeviceMemory && mSizeInBytes > 0);
    SYRINX_CUDA_ASSERT(cudaFree(mDeviceMemory));
    mDeviceMemory = nullptr;
    mSizeInBytes = 0;
}


void CudaBuffer::upload(const uint8_t *src, size_t writeOffset, size_t writeSizeInBytes)
{
    SYRINX_ASSERT(src && mDeviceMemory != nullptr);
    SYRINX_ASSERT(writeOffset + writeSizeInBytes <= mSizeInBytes);

    auto writeDst = reinterpret_cast<void*>(mDeviceMemory + writeOffset);
    SYRINX_CUDA_ASSERT(cudaMemcpy(writeDst, src, writeSizeInBytes, cudaMemcpyHostToDevice));
}


uint8_t* CudaBuffer::download(uint8_t *dest, size_t readOffset, size_t readSizeInBytes)
{
    SYRINX_ASSERT(dest && mDeviceMemory != nullptr);
    SYRINX_ASSERT(readOffset + readSizeInBytes <= mSizeInBytes);

    auto readSrc = reinterpret_cast<void*>(mDeviceMemory + readOffset);
    SYRINX_CUDA_ASSERT(cudaMemcpy(dest, readSrc, readSizeInBytes, cudaMemcpyDeviceToHost));
    return dest;
}

} // namespace Syrinx
