#pragma once
#include <cstdint>
#include <vector>
#include <better-enums/enum.h>
#include "SyrinxHardwareVertexBuffer.h"

namespace Syrinx {

BETTER_ENUM(VertexAttributeSemantic, uint8_t,
    Undefined,
    Position,
    Normal,
    TexCoord,
    Tangent,
    Bitangent,
    Color
);


BETTER_ENUM(VertexAttributeDataType, uint8_t,
    Undefined,
    UBYTE1,
    UBYTE2,
    UBYTE3,
    UBYTE4,

    SHORT1,
    SHORT2,
    SHORT3,
    SHORT4,

    FLOAT1,
    FLOAT2,
    FLOAT3,
    FLOAT4,

    DOUBLE1,
    DOUBLE2,
    DOUBLE3,
    DOUBLE4
);


using VertexAttributeLocation = uint8_t;
using VertexBufferBindingPoint = uint8_t;


class VertexAttributeDescription {
public:
    static uint32_t getByteSizeForDataType(VertexAttributeDataType dataType);

public:
    VertexAttributeDescription() = default;
    bool operator==(const VertexAttributeDescription& rhs) const;

    VertexAttributeDescription& setSemantic(const VertexAttributeSemantic& semantic);
    VertexAttributeDescription& setDataType(const VertexAttributeDataType& dataType);
    VertexAttributeDescription& setLocation(const VertexAttributeLocation& location);
    VertexAttributeDescription& setBindingPoint(const VertexBufferBindingPoint& bindingPoint);
    VertexAttributeDescription& setDataOffset(size_t dataOffset);
    VertexAttributeSemantic getSemantic() const;
    VertexAttributeLocation getLocation() const;
    VertexAttributeDataType getDataType() const;
    VertexBufferBindingPoint getBindingPoint() const;
    size_t getDataOffset() const;

private:
    VertexAttributeSemantic mSemantic = VertexAttributeSemantic::Position;
    VertexAttributeDataType mDataType = VertexAttributeDataType::FLOAT3;
    VertexAttributeLocation mLocation = 0;
    VertexBufferBindingPoint mBindingPoint = 0;
    size_t mDataOffset = 0;
};

} // namespace Syrinx