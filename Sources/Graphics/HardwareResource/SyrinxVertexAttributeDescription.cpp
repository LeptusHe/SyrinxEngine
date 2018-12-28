#include "HardwareResource/SyrinxVertexAttributeDescription.h"
#include <Common/SyrinxAssert.h>
#include <Logging/SyrinxLogManager.h>

namespace Syrinx {

uint32_t VertexAttributeDescription::getByteSizeForDataType(Syrinx::VertexAttributeDataType dataType)
{
    switch (dataType) {
        case Syrinx::VertexAttributeDataType::SHORT1: return 1 * sizeof(short);
        case Syrinx::VertexAttributeDataType::SHORT2: return 2 * sizeof(short);
        case Syrinx::VertexAttributeDataType::SHORT3: return 3 * sizeof(short);
        case Syrinx::VertexAttributeDataType::SHORT4: return 4 * sizeof(short);

        case Syrinx::VertexAttributeDataType::FLOAT1: return 1 * sizeof(float);
        case Syrinx::VertexAttributeDataType::FLOAT2: return 2 * sizeof(float);
        case Syrinx::VertexAttributeDataType::FLOAT3: return 3 * sizeof(float);
        case Syrinx::VertexAttributeDataType::FLOAT4: return 4 * sizeof(float);

        case Syrinx::VertexAttributeDataType::DOUBLE1: return 1 * sizeof(double);
        case Syrinx::VertexAttributeDataType::DOUBLE2: return 2 * sizeof(double);
        case Syrinx::VertexAttributeDataType::DOUBLE3: return 3 * sizeof(double);
        case Syrinx::VertexAttributeDataType::DOUBLE4: return 4 * sizeof(double);
        default: {
            SYRINX_ERROR_FMT("fail to get byte size for vertex data type [{}]", dataType._to_string());
            SYRINX_ASSERT(false && "fail to get byte size for vertex data type");
        }
    }
    return 0;
}


VertexAttributeDescription::VertexAttributeDescription(VertexAttributeBindingPoint bindingPoint,
                                                       VertexAttributeSemantic semantic,
                                                       VertexAttributeDataType dataType)
    : mBindingPoint(bindingPoint)
    , mSemantic(semantic)
    , mDataType(dataType)
{

}


bool VertexAttributeDescription::operator==(const VertexAttributeDescription& rhs) const
{
    return mSemantic == rhs.mSemantic && mDataType == rhs.mDataType;
}


VertexAttributeBindingPoint VertexAttributeDescription::getBindingPoint() const
{
    return mBindingPoint;
}


VertexAttributeSemantic VertexAttributeDescription::getSemantic() const
{
    return mSemantic;
}


VertexAttributeDataType VertexAttributeDescription::getDataType() const
{
    return mDataType;
}

} // namespace Syrinx
