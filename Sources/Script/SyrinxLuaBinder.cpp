#include "SyrinxLuaBinder.h"

namespace Syrinx {

void LuaBinder::bindToScript(sol::table& library)
{
    BasicTypeBind::bind(library);
    RenderStateBind::bind(library);
    RenderContextBind::bind(library);
}

} // namespace Syrinx
