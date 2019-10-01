#pragma once
#include <Script/SyrinxLuaCommon.h>
#include <Scene/SyrinxEntity.h>
#include <Scene/Component/SyrinxCamera.h>

namespace Syrinx {

class EntityComponentBind {
public:
    static void bind(sol::table& library)
    {
        bindEntity(library);
        bindCameraComponent(library);
    }

private:
    static void bindEntity(sol::table& library)
    {
        library.new_usertype<Entity>("Entity",
                                     "getName", &Entity::getName,
                                     "getController", &Entity::addController);
    }

    static void bindCameraComponent(sol::table& library)
    {
        library.new_usertype<Camera>("Camera",
                sol::constructors<Camera(std::string)>(),
                "getName", &Camera::getName);
    }
};

} // namespace Syrinx