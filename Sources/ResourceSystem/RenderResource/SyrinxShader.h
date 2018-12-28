#pragma once
#include "RenderResource/SyrinxRenderResource.h"
#include "RenderResource/SyrinxShaderParameter.h"
#include "RenderResource/SyrinxShaderPass.h"

namespace Syrinx {

class Shader : public RenderResource {
public:
    using ShaderPassMap = std::unordered_map<std::string, std::unique_ptr<ShaderPass>>;
    using ShaderPassList = std::vector<ShaderPass*>;
    using ShaderParameterMap = std::unordered_map<std::string, std::unique_ptr<ShaderParameter>>;
    using ShaderParameterList = std::vector<ShaderParameter*>;

public:
    explicit Shader(const std::string& name);
    ~Shader() override = default;

    void addShaderPassSet(std::vector<std::unique_ptr<ShaderPass>>&& shaderPassSet);
    void addShaderPass(std::unique_ptr<ShaderPass>&& shaderPass);
    void addShaderParameterSet(std::vector<std::unique_ptr<ShaderParameter>>&& shaderParameterSet);
    void addShaderParameter(std::unique_ptr<ShaderParameter>&& shaderParameter);
    ShaderPass* getShaderPass(const std::string& name) const;
    ShaderParameter* getShaderParameter(const std::string& name) const;
    const ShaderPassList& getShaderPassList() const;
    const ShaderParameterList& getShaderParameterList() const;

private:
    ShaderPassMap mShaderPassMap;
    ShaderPassList mShaderPassList;
    ShaderParameterMap mShaderParameterMap;
    ShaderParameterList mShaderParameterList;
};

} // namespace Syrinx