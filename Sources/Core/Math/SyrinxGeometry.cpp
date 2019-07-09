#include "SyrinxGeometry.h"
#include <cstdint>

namespace Syrinx {

template struct Offset2D<uint32_t>;
template struct Extent2D<uint32_t>;
template struct Rect2D<uint32_t>;

} // namespace Syrinx
