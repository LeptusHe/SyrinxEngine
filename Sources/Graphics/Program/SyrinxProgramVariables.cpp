#include "SyrinxProgramVariables.h"
#include "Manager/SyrinxHardwareResourceManager.h"

namespace Syrinx {

ProgramVars::ProgramVars(const ProgramReflector& programReflector)
    : mReflector(programReflector)
    , mUniformBufferList()
    , mSampledTextureList()
{
    SYRINX_ENSURE(mUniformBufferList.empty());
    SYRINX_ENSURE(mSampledTextureList.empty());

    for (const auto& textureInfo : programReflector.getSampledTextureList()) {
        mSampledTextureList.push_back({textureInfo, nullptr});
    }

    for (const auto& uniformBufferInfo : programReflector.getUniformBufferList()) {
        SYRINX_ASSERT(uniformBufferInfo);
        auto uboInfo = new UniformBufferInfo();
        *uboInfo = *uniformBufferInfo;
        auto *buffer = new uint8_t[uboInfo->size];
        std::memset(buffer, 0, uboInfo->size);
        uboInfo->buffer = buffer;
        SYRINX_ASSERT(uboInfo->buffer);
        uboInfo->setUniformBufferInfo(uboInfo);
        mUniformBufferList.push_back(uboInfo);
    }
}


ProgramVars::~ProgramVars()
{
    for (auto& uniformBufferInfo : mUniformBufferList) {
        delete [] uniformBufferInfo->buffer;
        delete uniformBufferInfo;
    }
}


std::string ProgramVars::getProgramName() const
{
    return mReflector.getProgramName();
}


void ProgramVars::setTexture(const std::string& texName, const SampledTexture *sampledTexture)
{
    SYRINX_EXPECT(sampledTexture);
    for (auto& textureVariable : mSampledTextureList) {
        if (textureVariable.first->name == texName) {
            textureVariable.second = sampledTexture;
            return;
        }
    }
    SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "fail to set texture [{}] because of not exist", texName);
}


bool ProgramVars::hasUniformBuffer(const std::string& name) const
{
    for (const auto& uniformBufferInfo : mUniformBufferList) {
        if (uniformBufferInfo->name == name) {
            return true;
        }
    }
    return false;
}


bool ProgramVars::hasTexture(const std::string& name) const
{
    for (const auto& textureInfo : mSampledTextureList) {
        if (textureInfo.first->name == name) {
            return true;
        }
    }
    return false;
}


UniformBufferInfo* ProgramVars::getUniformBuffer(const std::string& name)
{
    for (const auto& uniformBufferInfo : mUniformBufferList) {
        if (uniformBufferInfo->name == name) {
            return uniformBufferInfo;
        }
    }
    return nullptr;
}


const TextureInfo* ProgramVars::getTextureInfo(const std::string& name)
{
    for (const auto& sampledTexture : mSampledTextureList) {
        if (sampledTexture.first->name == name) {
            return sampledTexture.first;
        }
    }
    return nullptr;
}


const ProgramVars::UniformBufferList& ProgramVars::getUniformBufferList() const
{
    return mUniformBufferList;
}


const ProgramVars::SampledTextureList& ProgramVars::getSampledTextureList() const
{
    return mSampledTextureList;
}


UniformBufferInfo& ProgramVars::operator[](const std::string& name)
{
    for (auto& uniformBufferInfo : mUniformBufferList) {
        if (uniformBufferInfo->name == name) {
            return *uniformBufferInfo;
        }
    }
    SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                               "fail to get uniform buffer [{}] in program [{}] because it doesn't exist",
                               name, mReflector.getProgramName());
    SHOULD_NOT_GET_HERE();
}

} // namespace Syrinx
