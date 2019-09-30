#pragma once
#include <Script/SyrinxLuaScript.h>
#include <Pipeline/SyrinxRenderPass.h>

namespace Syrinx {

class LuaScriptBinding {
public:
    void exportLibrary();
    void run(const std::string& script);

private:
    void exportRenderContextClass();
    void exportEntityRendererClass();

private:
    sol::state mLuaVM;
    sol::table mLibrary;
};

} // namespace Syrinx