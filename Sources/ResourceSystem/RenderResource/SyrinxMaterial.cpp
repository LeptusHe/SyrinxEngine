#include "RenderResource/SyrinxMaterial.h"

namespace Syrinx {

Material::Material(const std::string& mName)
    : RenderResource(mName)
    , mShader(nullptr)
{
    SYRINX_ENSURE(!mShader);
}


void Material::setShader(Shader *shader)
{
    mShader = shader;
    SYRINX_ENSURE(mShader);
}


void Material::addMaterialParameter(ShaderParameter *shaderParameter)
{
    SYRINX_EXPECT(shaderParameter);
    SYRINX_EXPECT(!getMaterialParameter(shaderParameter->getName()));
    mParameterList.push_back(shaderParameter);
    mParameterMap[shaderParameter->getName()] = std::unique_ptr<ShaderParameter>(shaderParameter);
    SYRINX_ENSURE(getMaterialParameter(shaderParameter->getName()) == shaderParameter);
}


Shader* Material::getShader() const
{
    return mShader;
}


ShaderParameter* Material::getMaterialParameter(const std::string& name) const
{
    SYRINX_EXPECT(!name.empty());
    auto iter = mParameterMap.find(name);
    if  (iter == std::end(mParameterMap)) {
        return nullptr;
    }
    return iter->second.get();
}


const Material::MaterialParameterList& Material::getMaterialParameterList() const
{
    return mParameterList;
}

} // namespace Syrinx