#pragma once
#include <better-enums/enum.h>

namespace Syrinx {

BETTER_ENUM(MaterialType, uint8_t, Matte)

class RadianceMaterial {
public:


private:
    MaterialType mType;
};

} // namespace Syrinx