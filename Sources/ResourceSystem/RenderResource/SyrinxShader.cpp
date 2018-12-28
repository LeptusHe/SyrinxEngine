#include "RenderResource/SyrinxShader.h"

namespace Syrinx {

Shader::Shader(const std::string& name) : RenderResource(name)
{

}


void Shader::addShaderPassSet(std::vector<std::unique_ptr<Syrinx::ShaderPass>>&& shaderPassSet)
{
    for (auto& shaderPass : shaderPassSet) {
        addShaderPass(std::move(shaderPass));
    }
}


void Shader::addShaderPass(std::unique_ptr<ShaderPass>&& shaderPass)
{
    SYRINX_EXPECT(shaderPass);
    mShaderPassList.push_back(shaderPass.get());
    mShaderPassMap[shaderPass->getName()] = std::move(shaderPass);
    SYRINX_ENSURE(!shaderPass);
    SYRINX_ENSURE(getShaderPass(mShaderPassList[mShaderPassList.size() - 1]->getName()));
}


void Shader::addShaderParameterSet(std::vector<std::unique_ptr<Syrinx::ShaderParameter>>&& shaderParameterSet)
{
    for (auto& shaderParameter : shaderParameterSet) {
        addShaderParameter(std::move(shaderParameter));
    }
}


void Shader::addShaderParameter(std::unique_ptr<ShaderParameter>&& shaderParameter)
{
    SYRINX_EXPECT(shaderParameter);
    mShaderParameterList.push_back(shaderParameter.get());
    mShaderParameterMap[shaderParameter->getName()] = std::move(shaderParameter);
    SYRINX_ENSURE(!shaderParameter);
    SYRINX_ENSURE(getShaderParameter(mShaderParameterList[mShaderParameterList.size() - 1]->getName()));
}


ShaderPass* Shader::getShaderPass(const std::string& name) const
{
    const auto& iter = mShaderPassMap.find(name);
    return (iter != std::end(mShaderPassMap)) ? iter->second.get() : nullptr;
}


ShaderParameter* Shader::getShaderParameter(const std::string& name) const
{
    const auto& iter = mShaderParameterMap.find(name);
    return (iter != std::end(mShaderParameterMap)) ? iter->second.get() : nullptr;
}


const Shader::ShaderPassList& Shader::getShaderPassList() const
{
    return mShaderPassList;
}


const Shader::ShaderParameterList& Shader::getShaderParameterList() const
{
    return mShaderParameterList;
}

} // namespace
