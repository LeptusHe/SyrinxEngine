#pragma once
#include <Script/SyrinxLuaCommon.h>
#include <Common/SyrinxAssert.h>

namespace Syrinx {

class RenderContextBind {
public:
    static void bind(sol::table& library)
    {
        SYRINX_EXPECT(library);

        bindRenderContext(library);
    }


private:
    static void bindRenderContext(sol::table& library)
    {
        library.new_usertype<RenderContext>("RenderContext",
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
};


} // namespace Syrinx