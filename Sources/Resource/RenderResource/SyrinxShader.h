#pragma once
#include "RenderResource/SyrinxResource.h"
#include <HardwareResource/SyrinxProgramStage.h>
#include <HardwareResource/SyrinxProgramPipeline.h>

namespace Syrinx {

class Shader : public Resource {
public:
    using ShaderModuleType = int;
    using ShaderModuleMap = std::unordered_map<ShaderModuleType, ProgramStage*>;

public:
    explicit Shader(const std::string& name);
    ~Shader() override = default;

    void addProgramPipeline(ProgramPipeline *programPipeline);
    ProgramStage* getShaderModule(const ProgramStageType& type) const;
    ProgramPipeline* getProgramPipeline() const;

private:
    ShaderModuleMap mShaderModuleMap;
    ProgramPipeline *mProgramPipeline;
};

} // namespace Syrinx