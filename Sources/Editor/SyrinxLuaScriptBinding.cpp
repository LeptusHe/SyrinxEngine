#include "SyrinxLuaScriptBinding.h"
#include <Graphics/SyrinxRenderContext.h>
#include <Pipeline/SyrinxEntityRenderer.h>

namespace Syrinx {

void LuaScriptBinding::exportLibrary()
{
    SYRINX_EXPECT(!mLibrary);

    mLuaVM.open_libraries();
    mLibrary = mLuaVM["syrinx"].get_or_create<sol::table>();

    exportRenderContextClass();
    exportEntityRendererClass();
}


void LuaScriptBinding::exportRenderContextClass()
{
    mLibrary.new_usertype<RenderContext>("RenderContext",
            "pushRenderState", &RenderContext::pushRenderState,
            "popRenderState", &RenderContext::popRenderState,
            "clearRenderTarget", &RenderContext::clearRenderTarget,
            "clearDepth", &RenderContext::clearDepth,
            "setRenderState", &RenderContext::setRenderState,
            "setColorBlendState", &RenderContext::setColorBlendState,
            "setCullState", &RenderContext::setCullState,
            "setDepthState", &RenderContext::setDepthState,
            "getRenderState", &RenderContext::getRenderState);
}


void LuaScriptBinding::exportEntityRendererClass()
{
    mLibrary.new_usertype<EntityRenderer>("EntityRenderer",
            "render", &EntityRenderer::render);
}

} // namespace Syrinx
