#include "HardwareResource/SyrinxProgramStage.h"
#include <Common/SyrinxAssert.h>
#include <Logging/SyrinxLogManager.h>
#include "SyrinxConstantTranslator.h"
#include "Manager/SyrinxHardwareResourceManager.h"

namespace Syrinx {

bool ProgramStage::sameType(const ProgramStage& lhs, const ProgramStage& rhs)
{
    return lhs.mType == rhs.mType;
}


ProgramStage::ProgramStage(const std::string& name, HardwareResourceManager *hardwareResourceManager)
    : HardwareResource(name)
    , mHardwareResourceManager(hardwareResourceManager)
    , mBinarySource()
    , mType(ProgramStageType::UndefinedStage)
    , mReflector(nullptr)
    , mIsHardwareUniformBufferCreated(false)
    , mProgramVarsList()
    , mProgramVars(nullptr)
{
    SYRINX_ENSURE(mHardwareResourceManager);
    SYRINX_ENSURE(mBinarySource.empty());
    SYRINX_ENSURE(mType._value == ProgramStageType::UndefinedStage);
    SYRINX_ENSURE(!mReflector);
    SYRINX_ENSURE(!mIsHardwareUniformBufferCreated);
    SYRINX_ENSURE(mProgramVarsList.empty());
    SYRINX_ENSURE(!mProgramVars);
}


bool ProgramStage::operator<(const ProgramStage& rhs)
{
    return mType < rhs.mType;
}


void ProgramStage::setBinarySource(const std::vector<uint32_t>& binarySource)
{
    SYRINX_EXPECT(!binarySource.empty());
    mBinarySource = binarySource;
    SYRINX_ENSURE(!mBinarySource.empty());
}


void ProgramStage::setBinarySource(std::vector<uint32_t>&& binarySource)
{
    SYRINX_EXPECT(!binarySource.empty());
    mBinarySource = std::move(binarySource);
    SYRINX_ENSURE(!mBinarySource.empty());
    SYRINX_ENSURE(binarySource.empty());
}


void ProgramStage::setType(ProgramStageType type)
{
    SYRINX_EXPECT(type._value != ProgramStageType::UndefinedStage);
    mType = type;
    SYRINX_ENSURE(mType._value != ProgramStageType::UndefinedStage);
}


ProgramStageType ProgramStage::getType() const
{
    return mType;
}


bool ProgramStage::create()
{
    SYRINX_EXPECT(!isCreated());
    SYRINX_EXPECT(isValidToCreate());
    SYRINX_EXPECT(!mBinarySource.empty());

    GLuint shader = glCreateShader(ConstantTranslator::getProgramStageType(getType()));
    glShaderBinary(1, &shader, GL_SHADER_BINARY_FORMAT_SPIR_V, mBinarySource.data(), sizeof(mBinarySource[0]) * mBinarySource.size());
    glSpecializeShader(shader, "main", 0, nullptr, nullptr);

    constexpr int maxInfoLogSize = 512;
    char infoLog[maxInfoLogSize];
    GLint infoLogLength = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);
    GLint succeedToCompile = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &succeedToCompile);
    if (!succeedToCompile || infoLogLength > 0) {
        glGetShaderInfoLog(shader, maxInfoLogSize, nullptr, infoLog);
        if (!succeedToCompile) {
            SYRINX_ERROR_FMT("fail to create shader for program stage[name={}] because [{}]", getName(), std::string(infoLog));
            glDeleteShader(shader);
            return false;
        } else {
            SYRINX_DEBUG_FMT("shader compile log for program [{}]: {}", getName(), std::string(infoLog));
        }
    }

    GLuint programStageHandle = glCreateProgram();
    glProgramParameteri(programStageHandle, GL_PROGRAM_SEPARABLE, GL_TRUE);
    glAttachShader(programStageHandle, shader);
    glLinkProgram(programStageHandle);

    GLint succeedToLink = false;
    glGetProgramiv(programStageHandle, GL_LINK_STATUS, &succeedToLink);
    glGetProgramiv(programStageHandle, GL_INFO_LOG_LENGTH, &infoLogLength);
    if (!succeedToLink || infoLogLength > 0) {
        glGetProgramInfoLog(programStageHandle, maxInfoLogSize, nullptr, infoLog);
        if (!succeedToLink) {
            SYRINX_ERROR_FMT("fail to create program stage[name={}] because [{}]", getName(), std::string(infoLog));
            glDetachShader(programStageHandle, shader);
            glDeleteShader(shader);
            glDeleteProgram(programStageHandle);
            return false;
        } else {
            SYRINX_ERROR_FMT("link program stage[name={}] log: [{}]", getName(), std::string(infoLog));
        }
    }
    glDetachShader(programStageHandle, shader);
    glDeleteShader(shader);
    setHandle(programStageHandle);
    SYRINX_ENSURE(isCreated());
    return true;
}


ProgramVars* ProgramStage::getProgramVars()
{
    SYRINX_EXPECT(isCreated());
    if (!mIsHardwareUniformBufferCreated) {
        SYRINX_EXPECT(!mProgramVars);
        mReflector = std::make_unique<ProgramReflector>(getName(), std::move(mBinarySource));
        createHardwareUniformBuffer();
        mProgramVars = new ProgramVars(*mReflector);
        mProgramVarsList.push_back(std::unique_ptr<ProgramVars>(mProgramVars));
    }
    auto programVars = new ProgramVars(*mReflector);
    mProgramVarsList.push_back(std::unique_ptr<ProgramVars>(programVars));
    SYRINX_ENSURE(mProgramVars);
    return programVars;
}


HardwareUniformBuffer* ProgramStage::getHardwareUniformBuffer(const std::string& uniformBufferName) const
{
    const std::string& hardwareUniformBufferName = uniformBufferName;
    auto iter = mHardwareUniformBufferList.find(hardwareUniformBufferName);
    if (iter ==  std::end(mHardwareUniformBufferList)) {
        return nullptr;
    }
    return iter->second;
}


void ProgramStage::updateProgramVars(const ProgramVars& programVars)
{
    SYRINX_EXPECT(isCreated());
    SYRINX_ENSURE(mProgramVars);
    SYRINX_EXPECT(mIsHardwareUniformBufferCreated);
    if (programVars.getProgramName() != getName()) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                   "fail to update program var for program [{}] because the program vars doesn't belong to program [{}]",
                                   getName(), programVars.getProgramName());
    }

    for (const auto& sampledTexture : programVars.getSampledTextureList()) {
        const auto& texName = sampledTexture.first->name;
        mProgramVars->setTexture(texName, sampledTexture.second);
    }

    for (const auto& uniformBufferInfo : programVars.getUniformBufferList()) {
        SYRINX_ASSERT(uniformBufferInfo);
        const auto& uniformBufferName = uniformBufferInfo->name;
        auto& srcUniformBuffer = uniformBufferInfo->buffer;
        auto destUniformBufferInfo = mProgramVars->getUniformBuffer(uniformBufferInfo->name);
        auto destUniformBuffer = destUniformBufferInfo->buffer;
        SYRINX_ASSERT(uniformBufferInfo->size == destUniformBufferInfo->size);
        std::memcpy(destUniformBuffer, srcUniformBuffer, uniformBufferInfo->size);
    }
}


void ProgramStage::uploadParametersToGpu()
{
    for (const auto& uniformBufferInfo : mProgramVars->getUniformBufferList()) {
        auto hardwareUniformBuffer = mHardwareUniformBufferList[uniformBufferInfo->name];
        SYRINX_ASSERT(hardwareUniformBuffer);
        SYRINX_ASSERT(uniformBufferInfo->buffer);
        hardwareUniformBuffer->setData(0, uniformBufferInfo->buffer, uniformBufferInfo->size);
    }

    for (const auto& uniformBufferInfo : mHardwareUniformBufferList) {
        auto uniformBuffer = uniformBufferInfo.second;
        SYRINX_ASSERT(uniformBuffer);
        uniformBuffer->uploadToGpu();
    }
}


void ProgramStage::bindResources()
{
    SYRINX_EXPECT(isCreated());
    SYRINX_ENSURE(mProgramVars);
    SYRINX_EXPECT(mIsHardwareUniformBufferCreated);

    for (const auto& sampledTextureInfo : mProgramVars->getSampledTextureList()) {
        auto textureInfo = sampledTextureInfo.first;
        auto sampledTexture = sampledTextureInfo.second;
        SYRINX_ASSERT(textureInfo);

        if (sampledTexture) {
            const auto bindingPoint = textureInfo->binding;
            glBindTextureUnit(bindingPoint, sampledTexture.getTextureView().getHandle());
            glBindSampler(bindingPoint, sampledTexture.getSampler().getHandle());
        }
    }

    for (const auto& uniformBufferInfo : mProgramVars->getUniformBufferList()) {
        auto& hardwareUniformBuffer = mHardwareUniformBufferList[uniformBufferInfo->name];
        glBindBufferBase(GL_UNIFORM_BUFFER, uniformBufferInfo->binding, hardwareUniformBuffer->getHandle());
    }
}


void ProgramStage::updateParameter(const std::string& name, int value)
{
    updateParameter(name, [value, this](GLint paramLocation) {
        glProgramUniform1i(getHandle(), paramLocation, value);
    });
}


void ProgramStage::updateParameter(const std::string& name, GLuint64 value)
{
    updateParameter(name, [&value, this](GLint paramLocation) {
        glProgramUniformHandleui64ARB(getHandle(), paramLocation, value);
    });
}


void ProgramStage::updateParameter(const std::string& name, float value)
{
    updateParameter(name, [value, this](GLint paramLocation) {
        glProgramUniform1f(getHandle(), paramLocation, value);
    });
}


void ProgramStage::updateParameter(const std::string& name, const Color& color)
{
    updateParameter(name, static_cast<Vector4f>(color));
}


void ProgramStage::updateParameter(const std::string& name, const Vector2f& value)
{
    updateParameter(name, [&value, this](GLint paramLocation) {
        glProgramUniform2f(getHandle(), paramLocation, value.x, value.y);
    });
}


void ProgramStage::updateParameter(const std::string& name, const Vector3f& value)
{
    updateParameter(name, [&value, this](GLint paramLocation) {
        glProgramUniform3f(getHandle(), paramLocation, value.x, value.y, value.z);
    });
}


void ProgramStage::updateParameter(const std::string& name, const Vector4f& value)
{
    updateParameter(name, [&value, this](GLint paramLocation) {
        glProgramUniform4f(getHandle(), paramLocation, value.x, value.y, value.z, value.w);
    });
}


void ProgramStage::updateParameter(const std::string& name, const Matrix4x4& value)
{
    updateParameter(name, [&value, this](GLint paramLocation) {
        glProgramUniformMatrix4fv(getHandle(), paramLocation, 1, GL_FALSE, GetRawValue(value));
    });
}


GLint ProgramStage::getParameterLocation(const std::string& name) const
{
    SYRINX_EXPECT(isCreated());
    return glGetUniformLocation(getHandle(), name.c_str());
}


void ProgramStage::createHardwareUniformBuffer()
{
    SYRINX_EXPECT(mBinarySource.empty());
    SYRINX_EXPECT(mReflector);
    SYRINX_EXPECT(!mIsHardwareUniformBufferCreated);

    for (const auto& uniformBufferInfo : mReflector->getUniformBufferList()) {
        SYRINX_ASSERT(uniformBufferInfo);
        const std::string hardwareUniformBufferName = getHardwareUniformBufferName(uniformBufferInfo->name);
        size_t bufferSize = uniformBufferInfo->size;
        auto hardwareUniformBuffer = mHardwareResourceManager->createUniformBuffer(hardwareUniformBufferName, bufferSize);
        mHardwareUniformBufferList.insert({uniformBufferInfo->name, hardwareUniformBuffer});
    }
    mIsHardwareUniformBufferCreated = true;
    SYRINX_ENSURE(mIsHardwareUniformBufferCreated);
}


bool ProgramStage::isValidParameterLocation(GLint location) const
{
    return location >= 0;
}


bool ProgramStage::isValidToCreate() const
{
    return (mType._value != ProgramStageType::UndefinedStage) && (!mBinarySource.empty());
}


std::string ProgramStage::getHardwareUniformBufferName(const std::string& uniformBufferName) const
{
    return getName() + "." + uniformBufferName;
}

} // namespace Syrinx
