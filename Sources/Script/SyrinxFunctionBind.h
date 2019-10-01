#pragma once
#include <Script/SyrinxLuaCommon.h>
#include <Math/SyrinxGeometry.h>

namespace Syrinx {

class FunctionBind {
public:
    static void bind(sol::table& library)
    {
        library["createViewport"] = [](int xOffset, int yOffset, int width, int height) {
            Rect2D<uint32_t> viewport;
            viewport.offset = {static_cast<uint32_t>(xOffset), static_cast<uint32_t>(yOffset)};
            viewport.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
            return viewport;
        };
    }
};

} // namespace Syrinx