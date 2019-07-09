#pragma once
#include <Program/SyrinxProgramVariables.h>
#include "SyrinxShader.h"

namespace Syrinx {

class ShaderVars {
public:
    using ProgramVarsMap = std::unordered_map<int, ProgramVars*>;

public:
    explicit ShaderVars(Shader *shader);

    const Shader& getShader() const;
    ProgramVars *getProgramVars(const ProgramStageType& type);

private:
    Shader *mShader;
    ProgramVarsMap mProgramVarsMap;
};

} // namespace Syrinx