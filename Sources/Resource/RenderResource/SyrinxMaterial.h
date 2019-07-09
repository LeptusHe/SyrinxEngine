#pragma once
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include "RenderResource/SyrinxResource.h"
#include "RenderResource/SyrinxShaderVars.h"

namespace Syrinx {

class Material : public Resource {
public:
    using ShaderVarsMap = std::unordered_map<std::string, std::unique_ptr<ShaderVars>>;

public:
    explicit Material(const std::string& mName);
    ~Material() override = default;

    void addShaderVars(std::unique_ptr<ShaderVars>&& shaderVars);
    ShaderVars* getShaderVars(const std::string& shaderName);

private:
    ShaderVarsMap mShaderVarsMap;
};

} // namespace Syrinx