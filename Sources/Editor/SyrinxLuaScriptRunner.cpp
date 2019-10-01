#include "SyrinxLuaScriptRunner.h"
#include <Script/SyrinxLuaBinder.h>

namespace Syrinx {

LuaScriptRunner::LuaScriptRunner()
{
    openLibraries();
    bind();
}


void LuaScriptRunner::openLibraries()
{
    mLuaVM.open_libraries(
        sol::lib::base,
        sol::lib::math,
        sol::lib::string,
        sol::lib::package);
}


void LuaScriptRunner::bind()
{
    mLibrary = mLuaVM["syrinx"].get_or_create<sol::table>();
    LuaBinder::bindToScript(mLibrary);
    mLuaVM.script("syrinx.entityRenderer = syrinx.EntityRenderer.new()");
    SYRINX_ENSURE(mLibrary);
}


void LuaScriptRunner::run(const std::string& source)
{
    mLuaVM.script(source);
}

} // namespace Syrinx
