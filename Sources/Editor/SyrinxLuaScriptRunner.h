#pragma once
#include <Script/SyrinxLuaCommon.h>
#include <Pipeline/SyrinxRenderPass.h>

namespace Syrinx {

class LuaScriptRunner {
public:
    LuaScriptRunner();
    ~LuaScriptRunner() = default;

    void run(const std::string& source);
    template <typename T> T get(const std::string& name);
    template <typename T> void set(const std::string& name, T&& value);
    template <typename T> void bindToLibrary(const std::string& name, T&& value);
private:
    void openLibraries();
    void bind();

private:
    sol::state mLuaVM;
    sol::table mLibrary;
};


template <typename T>
T LuaScriptRunner::get(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    return mLuaVM[name];
}


template <typename T>
void LuaScriptRunner::set(const std::string& name, T&& value)
{
    mLuaVM[name] = value;
}


template <typename T>
void LuaScriptRunner::bindToLibrary(const std::string& name, T&& value)
{
    SYRINX_EXPECT(!name.empty());
    SYRINX_EXPECT(mLibrary);
    mLibrary[name] = value;
}

} // namespace Syrinx