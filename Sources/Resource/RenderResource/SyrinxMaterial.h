#pragma once
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include "RenderResource/SyrinxResource.h"
#include "RenderResource/SyrinxShader.h"
#include "RenderResource/SyrinxShaderParameter.h"

namespace Syrinx {

class Material : public Resource {
public:
    using MaterialParameterList = std::vector<ShaderParameter*>;
    using MaterialParameterMap = std::unordered_map<std::string, std::unique_ptr<ShaderParameter>>;

public:
    explicit Material(const std::string& mName);
    ~Material() override = default;

    void setShader(Shader *shader);
    void addMaterialParameter(ShaderParameter *shaderParameter);
    Shader* getShader() const;
    ShaderParameter* getMaterialParameter(const std::string& name) const;
    const MaterialParameterList& getMaterialParameterList() const;

private:
    Shader *mShader;
    MaterialParameterList mParameterList;
    MaterialParameterMap mParameterMap;
};

} // namespace Syrinx