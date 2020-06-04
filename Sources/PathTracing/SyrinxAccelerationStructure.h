#pragma once
#include <vector>
#include <memory>
#include <optix.h>
#include "SyrinxCudaBuffer.h"

namespace Syrinx {


class AccelerationStructure {
public:
    using SharedPtr = std::shared_ptr<AccelerationStructure>;
    enum class Type {
        Geometry,
        Instance
    };

public:
    AccelerationStructure(Type type, OptixTraversableHandle handle, CudaBuffer&& buffer)
	    : mType(type)
	    , mHandle(handle)
	    , mBuffer(std::move(buffer))
    { }
    AccelerationStructure(const AccelerationStructure&) = delete;
	AccelerationStructure(AccelerationStructure&& rhs) noexcept;
	Type getType() const { return mType; }
	~AccelerationStructure() = default;

public:
	OptixTraversableHandle mHandle = 0;
	Type mType;
	CudaBuffer mBuffer;
};	
	
} // namespace Syrinx::Radiance