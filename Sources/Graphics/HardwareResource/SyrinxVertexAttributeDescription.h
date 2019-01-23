#pragma once
#include <cstdint>
#include <vector>
#include <better-enums/enum.h>

namespace Syrinx {

BETTER_ENUM(VertexAttributeSemantic, uint8_t,
    Undefined,
    Position,
    Normal,
    TexCoord,
    Tangent,
    Bitangent
);


BETTER_ENUM(VertexAttributeDataType, uint8_t,
    Undefined,
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


using VertexAttributeBindingPoint = uint8_t;


class VertexAttributeDescription {
public:
    static uint32_t getByteSizeForDataType(VertexAttributeDataType dataType);

public:
    VertexAttributeDescription(VertexAttributeBindingPoint bindingPoint,
                               VertexAttributeSemantic semantic,
                               VertexAttributeDataType dataType);
    bool operator==(const VertexAttributeDescription& rhs) const;

    VertexAttributeSemantic getSemantic() const;
    VertexAttributeDataType getDataType() const;
    VertexAttributeBindingPoint getBindingPoint() const;

private:
    VertexAttributeSemantic mSemantic;
    VertexAttributeDataType mDataType;
    VertexAttributeBindingPoint mBindingPoint;
};

} // namespace Syrinx