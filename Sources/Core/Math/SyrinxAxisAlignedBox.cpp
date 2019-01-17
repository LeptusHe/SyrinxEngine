#include "Math/SyrinxAxisAlignedBox.h"
#include "Common/SyrinxAssert.h"

namespace Syrinx {

AxisAlignedBox::AxisAlignedBox(const Vector3f& minimum, const Vector3f& maximum)
    : mMinimum(minimum), mMaximum(maximum)
{

}


const Vector3f& AxisAlignedBox::getMinimum() const
{
    return mMinimum;
}


const Vector3f& AxisAlignedBox::getMaximum() const
{
    return mMaximum;
}


Point3f AxisAlignedBox::getCornerVertex(AxisAlignedBoxCornerIndex cornerIndex) const
{
    switch (cornerIndex) {
        case AxisAlignedBoxCornerIndex::FAR_LEFT_BOTTOM: return {mMinimum.x, mMinimum.y, mMinimum.z};
        case AxisAlignedBoxCornerIndex::FAR_LEFT_TOP: return {mMinimum.x, mMaximum.y, mMinimum.z};
        case AxisAlignedBoxCornerIndex::FAR_RIGHT_BOTTOM: return {mMaximum.x, mMinimum.y, mMinimum.z};
        case AxisAlignedBoxCornerIndex::FAR_RIGHT_TOP: return {mMaximum.x, mMaximum.y, mMinimum.z};
        case AxisAlignedBoxCornerIndex::NEAR_LEFT_BOTTOM: return {mMinimum.x, mMinimum.y, mMaximum.z};
        case AxisAlignedBoxCornerIndex::NEAR_LEFT_TOP: return {mMinimum.x, mMaximum.y, mMaximum.z};
        case AxisAlignedBoxCornerIndex::NEAR_RIGHT_BOTTOM: return {mMaximum.x, mMinimum.y, mMaximum.z};
        case AxisAlignedBoxCornerIndex::NEAR_RIGHT_TOP: return {mMaximum.x, mMaximum.y, mMaximum.z};
        default: SYRINX_ASSERT(false && "undefined corner index for axis aligned bounding box");
    }
    return {0.0f, 0.0f, 0.0f};
}

} // namespace Syrinx