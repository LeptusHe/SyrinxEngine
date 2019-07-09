#include "SyrinxShaderVars.h"

namespace Syrinx {

ShaderVars::ShaderVars(Shader *shader) : mShader(shader)
{
    SYRINX_ENSURE(mShader);
}


const Shader& ShaderVars::getShader() const
{
    SYRINX_EXPECT(mShader);
    return *mShader;
}


ProgramVars* ShaderVars::getProgramVars(const ProgramStageType& type)
{
    const auto iter = mProgramVarsMap.find(type._to_index());
    if (iter == std::end(mProgramVarsMap)) {
        auto program = mShader->getShaderModule(type);
        if (!program) {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                       "fail to get [{}] program vars for shader [{}]",
                                       type._to_string(), mShader->getName());
        }
        auto programVars = program->getProgramVars();
        SYRINX_ASSERT(programVars);
        mProgramVarsMap[type._to_index()] = programVars;
        return programVars;
    } else {
        return iter->second;
    }
}

} // namespace Syrinx
