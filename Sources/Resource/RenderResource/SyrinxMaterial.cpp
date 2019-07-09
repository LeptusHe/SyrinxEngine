#include "RenderResource/SyrinxMaterial.h"

namespace Syrinx {

Material::Material(const std::string& mName)
    : Resource(mName)
    , mShaderVarsMap()
{
    SYRINX_ENSURE(mShaderVarsMap.empty());
}


void Material::addShaderVars(std::unique_ptr<ShaderVars>&& shaderVars)
{
    const auto& shader = shaderVars->getShader();
    const std::string& shaderName = shader.getName();
    SYRINX_ASSERT(!shaderName.empty());

    if (getShaderVars(shaderName)) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                  "fail to add shader vars [{}] into material [{}]",
                                  shaderName, getName());
    }
    mShaderVarsMap[shaderName] = std::move(shaderVars);
}


ShaderVars* Material::getShaderVars(const std::string& shaderName)
{
    SYRINX_EXPECT(!shaderName.empty());
    auto iter = mShaderVarsMap.find(shaderName);
    if (iter == std::end(mShaderVarsMap)) {
        return nullptr;
    }
    return iter->second.get();
}

} // namespace Syrinx