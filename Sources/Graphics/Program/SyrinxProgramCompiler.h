#pragma once
#include <shaderc/shaderc.hpp>
#include "HardwareResource/SyrinxProgramStage.h"

namespace Syrinx {

class ProgramCompiler {
public:
    using IncludeHandler = shaderc::CompileOptions::IncluderInterface;
    using CompileOptions = shaderc::CompileOptions;

public:
    ProgramCompiler();
    ~ProgramCompiler() = default;
    void setIncluder(std::unique_ptr<IncludeHandler>&& includeHandler);
    std::vector<uint32_t> compile(const std::string& programName, const std::string& source, const ProgramStageType& stageType, CompileOptions&& compileOptions);

private:
    shaderc_shader_kind getShaderKindFromProgramType(const ProgramStageType& shaderType) const;
    CompileOptions buildOptions(CompileOptions&& compileOptions);

private:
    shaderc::Compiler mCompiler;
    std::unique_ptr<IncludeHandler> mIncludeHandler;
};

} // namespace Syrinx