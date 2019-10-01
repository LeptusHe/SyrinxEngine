#include "SyrinxLuaBinder.h"
#include "SyrinxBasicTypeBind.h"
#include "SyrinxRenderStateBind.h"
#include "SyrinxRenderContextBind.h"
#include "SyrinxEntityComponentBind.h"
#include "SyrinxEntityRendererBind.h"
#include "SyrinxSceneBind.h"
#include "SyrinxFunctionBind.h"

namespace Syrinx {

void LuaBinder::bindToScript(sol::table& library)
{
    BasicTypeBind::bind(library);
    EntityComponentBind::bind(library);
    RenderStateBind::bind(library);
    RenderContextBind::bind(library);
    RenderStateBind::bind(library);
    EntityRendererBind::bind(library);
    SceneBind::bind(library);
    FunctionBind::bind(library);
}

} // namespace Syrinx
