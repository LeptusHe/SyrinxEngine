#pragma once
#include <Script/SyrinxLuaCommon.h>
#include <Math/SyrinxColor.h>

namespace Syrinx {

class BasicTypeBind {
public:
    static void bind(sol::table& library)
    {
        bindColor(library);
    }

private:
    static void bindColor(sol::table& library)
    {
        library.new_usertype<Color>("Color",
            sol::constructors<Color(float, float, float)>(),
            "red", &Color::red,
            "green", &Color::green,
            "blue", &Color::blue,
            "alpha", &Color::alpha);
    }
};

} // namespace Syrinx