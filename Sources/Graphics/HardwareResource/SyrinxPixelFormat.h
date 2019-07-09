#pragma once
#include <better-enums/enum.h>

namespace Syrinx {

BETTER_ENUM(PixelFormat, uint8_t,
    RED8,
    RG8,
    RGB8,
    RGBA8,

    RED_INTEGER,
    RG_INTEGER,
    RGB_INTEGER,
    RGBA_INTEGER,

    REDF,
    RGF,
    RG16F,
    RGBF,
    RGBAF,
    RGB16F,

    DEPTH16,
    DEPTH24,
    DEPTH32F,
    STENCIL_INDEX,
    DEPTH_STENCIL,
    UNKNOWN
);

} // namespace Syrinx
