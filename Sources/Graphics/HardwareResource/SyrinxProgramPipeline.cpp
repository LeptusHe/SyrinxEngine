#include "HardwareResource/SyrinxProgramPipeline.h"
#include <Common/SyrinxAssert.h>
#include <Logging/SyrinxLogManager.h>
#include "SyrinxConstantTranslator.h"

namespace Syrinx {

ProgramPipeline::ProgramPipeline(const std::string& name)
    : HardwareResource(name)
    , mProgramStageMap()
{
    SYRINX_ENSURE(mProgramStageMap.empty());
}


bool ProgramPipeline::create()
{
    SYRINX_EXPECT(!isCreated());
    GLuint handle = 0;
    glCreateProgramPipelines(1, &handle);
    setHandle(handle);
    SYRINX_ENSURE(isCreated());
    return true;
}


void ProgramPipeline::bindProgramStage(ProgramStage *programStage)
{
    SYRINX_EXPECT(programStage && programStage->isCreated());
    SYRINX_EXPECT(isCreated());

    if (sameProgramStageExists(programStage->getType())) {
        auto iter = mProgramStageMap.find(programStage->getType());
        auto programStageType = iter->first;
        const auto& beforeProgramName = iter->second->getName();
        const auto& afterProgramName = programStage->getName();
        SYRINX_DEBUG_FMT("change program stage [{}] from [{}] to [{}] for program pipeline [{}]", programStageType._to_string(), beforeProgramName, afterProgramName, getName());
    }
    mProgramStageMap[programStage->getType()] = programStage;
    glUseProgramStages(getHandle(), ConstantTranslator::getProgramStageTypeBitfield(programStage->getType()) , programStage->getHandle());

    if (isValidToLink()) {
        checkLinkState();
    }
    SYRINX_ENSURE(sameProgramStageExists(programStage->getType()));
}


ProgramStage * ProgramPipeline::getProgramStage(ProgramStageType type)
{
    auto iter = mProgramStageMap.find(type);
    return (iter != std::end(mProgramStageMap)) ? iter->second : nullptr;
}


const ProgramPipeline::ProgramStageMap& ProgramPipeline::getProgramStageMap() const
{
    return mProgramStageMap;
}


bool ProgramPipeline::sameProgramStageExists(ProgramStageType stageType) const
{
    return mProgramStageMap.find(stageType) != std::end(mProgramStageMap);
}


bool ProgramPipeline::isValidToCreate() const
{
    return true;
}


bool ProgramPipeline::isValidToLink() const
{
    if (mProgramStageMap.find(ProgramStageType::ComputeStage) != std::end(mProgramStageMap) && mProgramStageMap.find(ProgramStageType::VertexStage) == std::end(mProgramStageMap) && mProgramStageMap.find(ProgramStageType::FragmentStage) == std::end(mProgramStageMap))
        return true;
    if (mProgramStageMap.find(ProgramStageType::VertexStage) == std::end(mProgramStageMap))
        return false;
    if (mProgramStageMap.find(ProgramStageType::FragmentStage) == std::end(mProgramStageMap) && mProgramStageMap.find(ProgramStageType::GeometryStage) == std::end(mProgramStageMap))
        return false;
    return true;
}


void ProgramPipeline::checkLinkState()
{
    SYRINX_EXPECT(isCreated());
    GLint logInfoLength = 0;
    glGetProgramPipelineiv(getHandle(), GL_LINK_STATUS, &logInfoLength);
    const int maxLogSize = 512;
    char info[maxLogSize];
    if (logInfoLength != 0) {
        glGetProgramPipelineInfoLog(getHandle(), maxLogSize, nullptr, info);
        SYRINX_ERROR_FMT("info for program pipeline[name={}, info={}]", getName(), std::string(info));
    }
}

} // namespace Syrinx
