#pragma once
#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include <better-enums/enum.h>
#include <Math/SyrinxMath.h>
#include <Logging/SyrinxLogManager.h>
#include "SyrinxGraphicsEnums.h"
#include "HardwareResource/SyrinxHardwareResource.h"
#include "HardwareResource/SyrinxHardwareUniformBuffer.h"
#include "Program/SyrinxProgramVariables.h"
#include "Program/SyrinxProgramReflector.h"

namespace Syrinx {

class HardwareResourceManager;


class ProgramStage : public HardwareResource {
public:
    bool sameType(const ProgramStage& lhs, const ProgramStage& rhs);

public:
    ProgramStage(const std::string& name, HardwareResourceManager *hardwareResourceManager);
    ~ProgramStage() override = default;
    bool operator<(const ProgramStage& rhs);
    void setBinarySource(const std::vector<uint32_t>& binarySource);
    void setBinarySource(std::vector<uint32_t>&& binarySource);
    void setType(ProgramStageType type);
    ProgramStageType getType() const;
    bool create() override;
    ProgramVars* getProgramVars();
    HardwareUniformBuffer* getHardwareUniformBuffer(const std::string& uniformBufferName) const;
    void updateProgramVars(const ProgramVars& programVars);
    void uploadParametersToGpu();
    void bindResources();
    void updateParameter(const std::string& name, int value);
    void updateParameter(const std::string& name, GLuint64 value);
    void updateParameter(const std::string& name, float value);
    void updateParameter(const std::string& name, const Color& color);
    void updateParameter(const std::string& name, const Vector2f& value);
    void updateParameter(const std::string& name, const Vector3f& value);
    void updateParameter(const std::string& name, const Vector4f& value);
    void updateParameter(const std::string& name, const Matrix4x4& value);
    int getParameterLocation(const std::string& name) const;
    bool isValidParameterLocation(GLint location) const;

private:
    void createHardwareUniformBuffer();
    template <typename T> void updateParameter(const std::string& name, const T& updateOperation);
    bool isValidToCreate() const override;
    std::string getHardwareUniformBufferName(const std::string& uniformBufferName) const;

private:
    HardwareResourceManager *mHardwareResourceManager;
    std::vector<uint32_t> mBinarySource;
    ProgramStageType mType;
    std::unique_ptr<ProgramReflector> mReflector;
    bool mIsHardwareUniformBufferCreated;
    std::unordered_map<std::string, HardwareUniformBuffer*> mHardwareUniformBufferList;
    std::vector<std::unique_ptr<ProgramVars>> mProgramVarsList;
    ProgramVars *mProgramVars;
};


template <typename T>
void ProgramStage::updateParameter(const std::string& name, const T& updateOperation)
{
    auto paramLocation = getParameterLocation(name);
    if (!isValidParameterLocation(paramLocation)) {
        SYRINX_DEBUG_FMT("fail to update parameter [{}] for [{}] program [{}] because it does not exist", name, getType()._to_string(), getName());
        return;
    }
    updateOperation(paramLocation);
}

} // namespace Syrinx
