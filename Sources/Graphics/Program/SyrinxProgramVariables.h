#pragma once
#include <vector>
#include <better-enums/enum.h>
#include <spirv_glsl.hpp>
#include "HardwareResource/SyrinxHardwareUniformBuffer.h"
#include "HardwareResource/SyrinxHardwareTexture.h"
#include "HardwareResource/SyrinxSampledTexture.h"
#include "Program/SyrinxProgramReflector.h"

namespace Syrinx {

class HardwareResourceManager;

class ProgramVars {
public:
    using UniformBufferVariable = UniformBufferInfo*;
    using TextureVariable = std::pair<const TextureInfo*, const SampledTexture*>;
    using UniformBufferList = std::vector<UniformBufferVariable>;
    using SampledTextureList = std::vector<TextureVariable>;

public:
    explicit ProgramVars(const ProgramReflector& programReflector);
    ProgramVars(const ProgramVars&) = delete;
    ProgramVars& operator=(const ProgramVars&) = delete;
    ~ProgramVars();

    std::string getProgramName() const;
    bool hasUniformBuffer(const std::string& name) const;
    bool hasTexture(const std::string& name) const;
    UniformBufferInfo* getUniformBuffer(const std::string& name);
    const TextureInfo* getTextureInfo(const std::string& name);
    const UniformBufferList& getUniformBufferList() const;
    const SampledTextureList& getSampledTextureList() const;
    void setTexture(const std::string& texName, const SampledTexture* sampledTexture);
    UniformBufferInfo& operator[](const std::string& name);

private:
    const ProgramReflector& mReflector;
    UniformBufferList mUniformBufferList;
    SampledTextureList mSampledTextureList;
};


} // namespace Syrinx