#pragma once
#include <vector>
#include <optix.h>
#include <Common/SyrinxAssert.h>
#include <RenderResource/SyrinxResource.h>

namespace Syrinx {

class CudaBuffer : public Resource {
public:
    CudaBuffer() : Resource("CUDA Buffer") {}
	explicit CudaBuffer(const std::string& name) : Resource(name) {}
	CudaBuffer(const CudaBuffer&) = delete;
	CudaBuffer(CudaBuffer&& rhs) noexcept;
	~CudaBuffer() override;

	void allocate(size_t sizeInBytes);
	void upload(const uint8_t *src, size_t writeOffset, size_t writeSizeInBytes);
	uint8_t* download(uint8_t *dest, size_t readOffset, size_t readSizeInBytes);
	void resize(size_t sizeInBytes);
	void free();
    template <typename T> void allocateAndUpload(const T *data, size_t count);
    template <typename T> void allocateAndUpload(const std::vector<T>& data);
    template <typename T> void upload(const T *src, size_t count);
    template <typename T> T* download(T *dest, size_t count);
    template <typename T> T* getDevicePtr() const;
    CUdeviceptr getDevicePtr() const { return reinterpret_cast<CUdeviceptr>(mDeviceMemory); }
    size_t getSize() const { return mSizeInBytes; }

private:
    size_t mSizeInBytes = 0;
    uint8_t *mDeviceMemory = nullptr;
};


template <typename T>
void CudaBuffer::allocateAndUpload(const T *data, size_t count)
{
    auto sizeInBytes = count * sizeof(T);
    allocate(sizeInBytes);
    upload(reinterpret_cast<const uint8_t*>(data), 0, sizeInBytes);
}


template <typename T>
void CudaBuffer::allocateAndUpload(const std::vector<T>& dataList)
{
    SYRINX_ASSERT(!dataList.empty());
    const auto src = dataList.data();
    allocateAndUpload(src, dataList.size());
}


template <typename T>
void CudaBuffer::upload(const T* src, size_t count)
{
    const auto sizeInBytes = count * sizeof(T);
    upload(reinterpret_cast<const uint8_t*>(src), 0, sizeInBytes);
}


template <typename T>
T* CudaBuffer::download(T *dest, size_t count)
{
    auto readSizeInBytes = count * sizeof(T);
    download(reinterpret_cast<uint8_t*>(dest), 0, readSizeInBytes);
    return dest;
}


template <typename T>
T* CudaBuffer::getDevicePtr() const
{
    return reinterpret_cast<T*>(mDeviceMemory);
}

} // namespace Syrinx