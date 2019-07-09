#include "HardwareResource/SyrinxVertexAttributeDescription.h"
#include <Common/SyrinxAssert.h>
#include <Logging/SyrinxLogManager.h>

namespace Syrinx {

uint32_t VertexAttributeDescription::getByteSizeForDataType(VertexAttributeDataType dataType)
{
    switch (dataType) {
        case VertexAttributeDataType::UBYTE1: return 1 * sizeof(uint8_t);
        case VertexAttributeDataType::UBYTE2: return 2 * sizeof(uint8_t);
        case VertexAttributeDataType::UBYTE3: return 3 * sizeof(uint8_t);
        case VertexAttributeDataType::UBYTE4: return 4 * sizeof(uint8_t);

        case VertexAttributeDataType::SHORT1: return 1 * sizeof(short);
        case VertexAttributeDataType::SHORT2: return 2 * sizeof(short);
        case VertexAttributeDataType::SHORT3: return 3 * sizeof(short);
        case VertexAttributeDataType::SHORT4: return 4 * sizeof(short);

        case VertexAttributeDataType::FLOAT1: return 1 * sizeof(float);
        case VertexAttributeDataType::FLOAT2: return 2 * sizeof(float);
        case VertexAttributeDataType::FLOAT3: return 3 * sizeof(float);
        case VertexAttributeDataType::FLOAT4: return 4 * sizeof(float);

        case VertexAttributeDataType::DOUBLE1: return 1 * sizeof(double);
        case VertexAttributeDataType::DOUBLE2: return 2 * sizeof(double);
        case VertexAttributeDataType::DOUBLE3: return 3 * sizeof(double);
        case VertexAttributeDataType::DOUBLE4: return 4 * sizeof(double);
        default: {
            SYRINX_ERROR_FMT("fail to get byte size for vertex data type [{}]", dataType._to_string());
            SYRINX_ASSERT(false && "fail to get byte size for vertex data type");
        }
    }
    return 0;
}


bool VertexAttributeDescription::operator==(const VertexAttributeDescription& rhs) const
{
    return mSemantic == rhs.mSemantic && mDataType == rhs.mDataType;
}


VertexAttributeDescription& VertexAttributeDescription::setSemantic(const VertexAttributeSemantic& semantic)
{
    mSemantic = semantic;
    return *this;
}


VertexAttributeDescription& VertexAttributeDescription::setDataType(const VertexAttributeDataType& dataType)
{
    mDataType = dataType;
    return *this;
}


VertexAttributeDescription& VertexAttributeDescription::setLocation(const VertexAttributeLocation& location)
{
    mLocation = location;
    return *this;
}


VertexAttributeDescription& VertexAttributeDescription::setBindingPoint(const VertexBufferBindingPoint& bindingPoint)
{
    mBindingPoint = bindingPoint;
    return *this;
}


VertexAttributeDescription& VertexAttributeDescription::setDataOffset(size_t dataOffset)
{
    mDataOffset = dataOffset;
    return *this;
}


VertexAttributeSemantic VertexAttributeDescription::getSemantic() const
{
    return mSemantic;
}


VertexAttributeLocation VertexAttributeDescription::getLocation() const
{
    return mLocation;
}


VertexBufferBindingPoint VertexAttributeDescription::getBindingPoint() const
{
    return mBindingPoint;
}


VertexAttributeDataType VertexAttributeDescription::getDataType() const
{
    return mDataType;
}


size_t VertexAttributeDescription::getDataOffset() const
{
    return mDataOffset;
}

} // namespace Syrinx
