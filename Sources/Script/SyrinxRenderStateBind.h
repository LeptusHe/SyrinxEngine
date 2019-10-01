#pragma once
#include <Script/SyrinxLuaCommon.h>
#include <Graphics/SyrinxRenderState.h>

namespace Syrinx {

class RenderStateBind {
public:
    static void bind(sol::table& library)
    {
        bindRenderState(library);
        bindInputAssemblyState(library);
        bindViewportState(library);
        bindRasterizationState(library);
        bindDepthStencilState(library);
    }

private:
    static void bindRenderState(sol::table& library)
    {
        library.new_usertype<RenderState>("RenderState",
            sol::default_constructor,
            "inputAssemblyState", &RenderState::inputAssemblyState,
            "viewportState", &RenderState::viewportState,
            "rasterizationState", &RenderState::rasterizationState,
            "depthStencilState", &RenderState::depthStencilState,
            "colorBlendState", &RenderState::colorBlendState);
    }


    static void bindInputAssemblyState(sol::table& library)
    {
        library.new_usertype<InputAssemblyState>("InputAssemblyState",
            "topology", &InputAssemblyState::topology);
    }


    static void bindViewportState(sol::table& library)
    {
        library.new_usertype<ViewportState>("ViewportState",
            "viewport", &ViewportState::viewport,
            "enableScissor", &ViewportState::enableScissor,
            "scissor", &ViewportState::scissor);
    }


    static void bindRasterizationState(sol::table& library)
    {
        library.new_usertype<RasterizationState>("RasterizationState",
            "polygonMode", &RasterizationState::polygonMode,
            "cullMode", &RasterizationState::cullMode);
    }


    static void bindDepthStencilState(sol::table& library)
    {
        library.new_usertype<DepthStencilState>("DepthStencilState",
            "enableDepthTest", &DepthStencilState::enableDepthTest,
            "enableDepthWrite", &DepthStencilState::enableDepthWrite);
    }

};

} // namespace Syrinx