#include "RenderResource/SyrinxShader.h"
#include <Exception/SyrinxException.h>

namespace Syrinx {

Shader::Shader(const std::string& name)
    : Resource(name)
    , mShaderModuleMap()
    , mProgramPipeline(nullptr)
{
    SYRINX_ENSURE(mShaderModuleMap.empty());
    SYRINX_ENSURE(!mProgramPipeline);
}


void Shader::addProgramPipeline(ProgramPipeline *programPipeline)
{
    SYRINX_EXPECT(programPipeline);
    mProgramPipeline = programPipeline;
}


ProgramStage* Shader::getShaderModule(const ProgramStageType& type) const
{
    return mProgramPipeline->getProgramStage(type);
}


ProgramPipeline* Shader::getProgramPipeline() const
{
    return mProgramPipeline;
}

} // namespace
