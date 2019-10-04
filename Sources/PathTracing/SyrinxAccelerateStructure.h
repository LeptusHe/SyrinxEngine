#pragma once
#include <vector>
#include <optix.h>
#include "SyrinxCudaBuffer.h"

namespace Syrinx {

class AccelerateStructure {
public:
	AccelerateStructure(OptixTraversableHandle handle, CudaBuffer&& buffer)
	    : mHandle(handle)
	    , mBuffer(std::move(buffer))
    { }
    AccelerateStructure(const AccelerateStructure&) = delete;
	AccelerateStructure(AccelerateStructure&& rhs) noexcept;
	~AccelerateStructure() = default;

private:
	OptixTraversableHandle mHandle = 0;
	CudaBuffer mBuffer;
};	
	
} // namespace Syrinx::Radiance