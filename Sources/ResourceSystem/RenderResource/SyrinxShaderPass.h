#pragma once
#include <string>
#include <unordered_map>
#include <HardwareResource/SyrinxVertexAttributeDescription.h>
#include <HardwareResource/SyrinxVertexDataDescription.h>
#include <HardwareResource/SyrinxProgramStage.h>
#include <HardwareResource/SyrinxProgramPipeline.h>
#include "RenderResource/SyrinxRenderResource.h"
#include "RenderResource/SyrinxShaderParameter.h"

namespace Syrinx {

class VertexAttribute {
public:
    VertexAttribute() = default;
    ~VertexAttribute() = default;

    void setName(const std::string& name);
    void setSemantic(const std::string& semantic) noexcept(false);
    void setDataType(const std::string& dataType) noexcept(false);
    const std::string& getName() const;
    VertexAttributeSemantic getSemantic() const;
    VertexAttributeDataType getDataType() const;
    bool isValid() const;

private:
    std::string mName;
    VertexAttributeSemantic mSemantic = VertexAttributeSemantic::Undefined;
    VertexAttributeDataType mDataType = VertexAttributeDataType::Undefined;
};




class ShaderPass : public RenderResource {
public:
    using ParameterMap = std::unordered_map<std::string, ShaderParameter*>;
    using ProgramStageList = std::vector<ProgramStage*>;
    using ProgramStageMap = std::unordered_map<uint8_t, ProgramStage*>;
    using ParameterRefMap = std::unordered_map<std::string, ProgramStageList>;
    using VertexAttributeList = std::vector<VertexAttribute>;

public:
    explicit ShaderPass(const std::string& name);
    ~ShaderPass() override = default;

    void addParameterRefSetForVertexProgram(const std::vector<ShaderParameter*>& parameterSet);
    void addParameterRefSetForFragmentProgram(const std::vector<ShaderParameter*>& parameterSet);
    void addVertexAttribute(const VertexAttribute& vertexAttribute);
    void addVertexAttributeSet(const std::vector<VertexAttribute>& vertexAttributeSet);
    void addProgramStage(ProgramStageType stageType, ProgramStage *programStage);
    void setProgramPipeline(ProgramPipeline *programPipeline);
    ShaderParameter* getParameter(const std::string& name) const;
    const ParameterMap& getParameterMap() const;
    const ParameterRefMap& getParameterReferenceMap() const;
    ProgramStageList* getProgramStageListForReferencedParameter(const std::string& parameterName);
    const ProgramStageList* getProgramStageListForReferencedParameter(const std::string& parameterName) const;
    const VertexAttributeList& getVertexAttributeList() const;
    const VertexAttribute* getVertexAttribute(const std::string& name) const;
    ProgramStage* getProgramStage(ProgramStageType stageType) const;
    ProgramPipeline* getProgramPipeline() const;

private:
    void addParameterRefSetForProgram(const std::vector<ShaderParameter*>& parameterSet, ProgramStage *programStage);
    void addParameter(ShaderParameter *parameter);

private:
    ParameterMap mParameterMap;
    ParameterRefMap mParameterRefMap;
    VertexAttributeList mVertexAttributeList;
    ProgramStageMap mProgramStageMap;
    ProgramPipeline *mProgramPipeline;
};

} // namespace Syrinx
