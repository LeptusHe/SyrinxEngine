#pragma once
#include <Script/SyrinxLuaCommon.h>
#include "SyrinxBasicTypeBind.h"
#include "SyrinxRenderStateBind.h"
#include "SyrinxRenderContextBind.h"

namespace Syrinx {

class LuaBinder {
public:
    static void bindToScript(sol::table& library);

};

} // namespace Syrinx