#pragma once
#include <HardwareResource/SyrinxProgramStage.h>

class DefaultProgramStage : public Syrinx::ProgramStage {
public:
    explicit DefaultProgramStage(const std::string& name) : ProgramStage(name) { }
    ~DefaultProgramStage() override = default;

    bool create() override
    {
        setState(Syrinx::HardwareResourceState::Created);
        return true;
    }

protected:
    bool isValidToCreate() const override { return true; }
};