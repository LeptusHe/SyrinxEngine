#pragma once
#include <map>
#include <memory>
#include "HardwareResource/SyrinxProgramStage.h"

namespace Syrinx {

class ProgramPipeline : public HardwareResource {
public:
    using ProgramStageMap = std::map<ProgramStageType, ProgramStage*>;

public:
    explicit ProgramPipeline(const std::string& name);
    ~ProgramPipeline() override = default;

    bool create() override;
    void bindProgramStage(ProgramStage *programStage);
    ProgramStage* getProgramStage(ProgramStageType type);
    const ProgramStageMap& getProgramStageMap() const;

private:
    bool sameProgramStageExists(ProgramStageType stageType) const;
    bool isValidToCreate() const override;
    bool isValidToLink() const;
    void checkLinkState();

private:
    ProgramStageMap mProgramStageMap;
};

} // namespace Syrinx
