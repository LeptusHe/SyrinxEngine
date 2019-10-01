#pragma once
#include <Script/SyrinxLuaCommon.h>
#include <Scene/SyrinxScene.h>

namespace Syrinx {

class SceneBind {
public:
    static void bind(sol::table& library)
    {
        bindScene(library);
        bindSceneNode(library);
    }

private:
    static void bindScene(sol::table& library)
    {
        library.new_usertype<Scene>("Scene",
                                    "getEntityList", &Scene::getEntityList,
                                    "getEntitiesWithRenderer", &Scene::getEntitiesWithComponent<Renderer>);
    }

    static void bindSceneNode(sol::table& library)
    {
        library.new_usertype<SceneNode>("SceneNode",
                "getName", &SceneNode::getName,
                "getEntity", &SceneNode::getEntity);
    }
};

} // namespace Syrinx