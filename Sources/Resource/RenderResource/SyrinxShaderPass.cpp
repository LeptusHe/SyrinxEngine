#include "RenderResource/SyrinxShaderPass.h"
#include <Common/SyrinxAssert.h>
#include <Container/String.h>
#include <Exception/SyrinxException.h>
#include <Logging/SyrinxLogManager.h>

namespace Syrinx {

void VertexAttribute::setName(const std::string& name)
{
    mName = name;
    SYRINX_ENSURE(!mName.empty());
}


void VertexAttribute::setSemantic(const std::string& semantic)
{
    SYRINX_EXPECT(!mName.empty());
    try {
        std::string semanticString = semantic;
        std::replace(std::begin(semanticString), std::end(semanticString), '-', ' ');
        auto tokenList = SplitStringBySpace(semanticString);
        semanticString = "";
        for (auto& token : tokenList) {
            token[0] = static_cast<char>(std::toupper(static_cast<int>(token[0])));
            semanticString += token;
        }
        mSemantic = VertexAttributeSemantic::_from_string(semanticString.c_str());
    } catch (std::exception& e) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "invalid semantic [{}] for input vertex attribute [{}]", semantic, mName);
    }
}


void VertexAttribute::setDataType(const std::string& dataType)
{
    SYRINX_EXPECT(!mName.empty());
    try {
        auto dataTypeString = ToUpper(dataType);
        mDataType = VertexAttributeDataType::_from_string(dataTypeString.c_str());
    } catch (std::exception& e) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "invalid data type [{}] for input vertex attribute [{}]", dataType, mName);
    }
}


const std::string& VertexAttribute::getName() const
{
    return mName;
}


VertexAttributeSemantic VertexAttribute::getSemantic() const
{
    return mSemantic;
}


VertexAttributeDataType VertexAttribute::getDataType() const
{
    return mDataType;
}


bool VertexAttribute::isValid() const
{
    return !mName.empty() &&
           (mSemantic._value != VertexAttributeSemantic::Undefined) &&
           (mDataType._value != VertexAttributeDataType::Undefined);
}


ShaderPass::ShaderPass(const std::string& name)
    : Resource(name)
    , mProgramPipeline(nullptr)
{
    SYRINX_ENSURE(!mProgramPipeline);
}


void ShaderPass::addParameterRefSetForVertexProgram(const std::vector<ShaderParameter*>& parameterSet)
{
    auto programStage = mProgramStageMap.find(ProgramStageType::VertexStage);
    if (programStage == std::end(mProgramStageMap)) {
        SYRINX_DEBUG_FMT("fail to add referenced parameter into vertex program for shader pass [{}] because it doesn't have vertex program", getName());
        return;
    }
    addParameterRefSetForProgram(parameterSet, programStage->second);
}


void ShaderPass::addParameterRefSetForFragmentProgram(const std::vector<ShaderParameter*>& parameterSet)
{
    auto programStage = mProgramStageMap.find(ProgramStageType::FragmentStage);
    if (programStage == std::end(mProgramStageMap)) {
        SYRINX_DEBUG_FMT("fail to add referenced parameter into fragment program for shader pass [{}] because it doesn't have fragment program", getName());
        return;
    }
    addParameterRefSetForProgram(parameterSet, programStage->second);
}


void ShaderPass::addParameterRefSetForProgram(const std::vector<ShaderParameter*>& parameterSet, ProgramStage *programStage)
{
    SYRINX_EXPECT(programStage);
    for (const auto referencedParam : parameterSet) {
        const std::string& name = referencedParam->getName();
        auto programStageList = getProgramStageListForReferencedParameter(name);
        if (!programStageList) {
            mParameterRefMap[name] = {programStage};
        } else {
            programStageList->push_back(programStage);
        }
        addParameter(referencedParam);
    }
}


void ShaderPass::addParameter(ShaderParameter *parameter)
{
    SYRINX_EXPECT(parameter);
    if (getParameter(parameter->getName())) {
        return;
    }
    mParameterMap[parameter->getName()] = parameter;
    SYRINX_ENSURE(getParameter(parameter->getName()));
}


void ShaderPass::addVertexAttribute(const VertexAttribute& vertexAttribute)
{
    SYRINX_EXPECT(vertexAttribute.isValid());
    mVertexAttributeList.push_back(vertexAttribute);
}


void ShaderPass::addVertexAttributeSet(const std::vector<VertexAttribute>& vertexAttributeSet)
{
    for (const auto& vertexAttribute : vertexAttributeSet) {
        addVertexAttribute(vertexAttribute);
    }
}


void ShaderPass::addProgramStage(ProgramStageType stageType, ProgramStage *programStage)
{
    SYRINX_EXPECT(programStage);
    if (auto oldProgramStage = getProgramStage(stageType); oldProgramStage) {
        SYRINX_DEBUG_FMT("change the [{}] program [{}] of shader pass [{}] to [{}]", stageType._to_string(), oldProgramStage->getName(), getName(), programStage->getName());
    }
    mProgramStageMap[stageType._value] = programStage;
    SYRINX_ENSURE(getProgramStage(stageType) == programStage);
}


void ShaderPass::setProgramPipeline(ProgramPipeline *programPipeline)
{
    SYRINX_EXPECT(programPipeline);
    mProgramPipeline = programPipeline;
    SYRINX_ENSURE(mProgramPipeline);
    SYRINX_ENSURE(mProgramPipeline == programPipeline);
}


ShaderParameter* ShaderPass::getParameter(const std::string& name) const
{
    const auto& iter = mParameterMap.find(name);
    return (iter != std::end(mParameterMap)) ? iter->second : nullptr;
}


const ShaderPass::ParameterMap& ShaderPass::getParameterMap() const
{
    return mParameterMap;
}


const ShaderPass::ParameterRefMap& ShaderPass::getParameterReferenceMap() const
{
    return mParameterRefMap;
}


ShaderPass::ProgramStageList* ShaderPass::getProgramStageListForReferencedParameter(const std::string& parameterName)
{
    SYRINX_EXPECT(!parameterName.empty());
    auto iter = mParameterRefMap.find(parameterName);
    if (iter == std::end(mParameterRefMap)) {
        return nullptr;
    }
    return &iter->second;
}


const ShaderPass::ProgramStageList* ShaderPass::getProgramStageListForReferencedParameter(const std::string& parameterName) const
{
    SYRINX_EXPECT(!parameterName.empty());
    auto iter = mParameterRefMap.find(parameterName);
    if (iter == std::end(mParameterRefMap)) {
        return nullptr;
    }
    return &iter->second;
}


const ShaderPass::VertexAttributeList& ShaderPass::getVertexAttributeList() const
{
    return mVertexAttributeList;
}


const VertexAttribute* ShaderPass::getVertexAttribute(const std::string& name) const
{
    for (const auto& vertexAttribute : mVertexAttributeList) {
        if (vertexAttribute.getName() == name) {
            return &vertexAttribute;
        }
    }
    return nullptr;
}


ProgramStage* ShaderPass::getProgramStage(ProgramStageType stageType) const
{
    auto iter = mProgramStageMap.find(stageType._value);
    if (iter == std::end(mProgramStageMap)) {
        return nullptr;
    }
    return iter->second;
}


ProgramPipeline* ShaderPass::getProgramPipeline() const
{
    return mProgramPipeline;
}

} // namespace Syrinx
