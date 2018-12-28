#pragma once
#include <memory>
#include <string>
#include <utility>
#include <better-enums/enum.h>
#include <Math/SyrinxMath.h>
#include <Logging/SyrinxLogManager.h>
#include "HardwareResource/SyrinxHardwareResource.h"

namespace Syrinx {

BETTER_ENUM(ProgramStageType, std::uint8_t,
    UndefinedStage,
    VertexStage,
    TessellationControlStage,
    TessellationEvaluationStage,
    GeometryStage,
    FragmentStage,
    ComputeStage
);


class ProgramStage : public HardwareResource {
public:
    bool sameType(const ProgramStage& lhs, const ProgramStage& rhs);

public:
    explicit ProgramStage(const std::string& name);
    ~ProgramStage() override = default;
    bool operator<(const ProgramStage& rhs);
    void setSource(const std::string& source);
    void setSource(std::string&& source);
    void setType(ProgramStageType type);
    const std::string& getSource() const;
    ProgramStageType getType() const;
    bool create() override;
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
    template <typename T> void updateParameter(const std::string& name, const T& updateOperation);
    bool isValidToCreate() const override;

private:
    std::string mSource;
    ProgramStageType mType;
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
