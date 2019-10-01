#pragma once
#include <Script/SyrinxLuaCommon.h>

namespace Syrinx {

class LuaBinder {
public:
    static void bindToScript(sol::table& library);
};

} // namespace Syrinx