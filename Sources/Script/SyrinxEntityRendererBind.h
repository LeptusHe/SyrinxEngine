#pragma once
#include <Script/SyrinxLuaCommon.h>
#include <Pipeline/SyrinxEntityRenderer.h>

namespace Syrinx {

class EntityRendererBind {
public:
    static void bind(sol::table& library)
    {
        library.new_usertype<EntityRenderer>("EntityRenderer",
                sol::default_constructor,
                "render", &EntityRenderer::render);
    }
};

} // namespace Syrinx