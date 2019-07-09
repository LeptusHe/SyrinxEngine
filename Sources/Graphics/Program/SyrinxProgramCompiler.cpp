#include "SyrinxProgramCompiler.h"
#include <Common/SyrinxAssert.h>
#include <Exception/SyrinxException.h>

namespace Syrinx {

ProgramCompiler::ProgramCompiler() : mCompiler(), mCompileOptions()
{
    mCompileOptions.SetForcedVersionProfile(450, shaderc_profile_core);
    mCompileOptions.SetTargetEnvironment(shaderc_target_env_opengl, 0);
    mCompileOptions.SetSourceLanguage(shaderc_source_language_glsl);
    mCompileOptions.SetWarningsAsErrors();
}


void ProgramCompiler::setIncluder(std::unique_ptr<ProgramCompiler::IncludeHandler>&& includeHandler)
{
    SYRINX_EXPECT(includeHandler);
    mCompileOptions.SetIncluder(std::move(includeHandler));
}


std::vector<uint32_t> ProgramCompiler::compile(const std::string& programName, const std::string& source, const ProgramStageType& stageType)
{
    shaderc_shader_kind shaderKind = getShaderKindFromProgramType(stageType);
    auto module = mCompiler.CompileGlslToSpv(source, shaderKind, programName.c_str(), mCompileOptions);
    if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "fail to compile program [{}] - {}", programName, module.GetErrorMessage());
    }
    return {module.begin(), module.end()};
}


shaderc_shader_kind ProgramCompiler::getShaderKindFromProgramType(const ProgramStageType& shaderType) const
{
#define stage_type_to_shader_kind(type, kind) if (shaderType._value == ProgramStageType::type) return kind

    stage_type_to_shader_kind(ProgramStageType::VertexStage, shaderc_glsl_vertex_shader);
    stage_type_to_shader_kind(ProgramStageType::TessellationControlStage, shaderc_glsl_tess_control_shader);
    stage_type_to_shader_kind(ProgramStageType::TessellationEvaluationStage, shaderc_glsl_tess_evaluation_shader);
    stage_type_to_shader_kind(ProgramStageType::GeometryStage, shaderc_glsl_geometry_shader);
    stage_type_to_shader_kind(ProgramStageType::FragmentStage, shaderc_glsl_fragment_shader);
    stage_type_to_shader_kind(ProgramStageType::ComputeStage, shaderc_glsl_compute_shader);
#undef stage_type_to_shader_kind

    SHOULD_NOT_GET_HERE();
}

} // namespace Syrinx
