#include "HardwareResource/SyrinxProgramStage.h"
#include <Common/SyrinxAssert.h>
#include <Logging/SyrinxLogManager.h>
#include "HardwareResource/SyrinxConstantTranslator.h"

namespace Syrinx {

bool ProgramStage::sameType(const ProgramStage& lhs, const ProgramStage& rhs)
{
    return lhs.mType == rhs.mType;
}


ProgramStage::ProgramStage(const std::string& name)
    : HardwareResource(name)
    , mSource()
    , mType(ProgramStageType::UndefinedStage)
{
    SYRINX_ENSURE(mSource.empty());
    SYRINX_ENSURE(mType._value == ProgramStageType::UndefinedStage);
}


bool ProgramStage::operator<(const ProgramStage& rhs)
{
    return mType < rhs.mType;
}


void ProgramStage::setSource(const std::string& source)
{
    SYRINX_EXPECT(!source.empty());
    mSource = source;
    SYRINX_ENSURE(!mSource.empty());
}


void ProgramStage::setSource(std::string&& source)
{
    SYRINX_EXPECT(!source.empty());
    mSource = source;
    SYRINX_ENSURE(!mSource.empty());
}


void ProgramStage::setType(ProgramStageType type)
{
    SYRINX_EXPECT(type._value != ProgramStageType::UndefinedStage);
    mType = type;
    SYRINX_ENSURE(mType._value != ProgramStageType::UndefinedStage);
}


const std::string& ProgramStage::getSource() const
{
    return mSource;
}


ProgramStageType ProgramStage::getType() const
{
    return mType;
}


bool ProgramStage::create()
{
    SYRINX_EXPECT(!isCreated());
    SYRINX_EXPECT(isValidToCreate());

    auto source = getSource().c_str();
    GLuint shader = glCreateShader(ConstantTranslator::getProgramStageType(getType()));
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLuint programStageHandle = glCreateProgram();
    glAttachShader(programStageHandle, shader);
    glProgramParameteri(programStageHandle, GL_PROGRAM_SEPARABLE, GL_TRUE);
    glLinkProgram(programStageHandle);

    //GLuint programStageHandle = glCreateShaderProgramv(ConstantTranslator::getProgramStageType(getType()), 1, &source);
    constexpr int maxInfoLogSize = 512;
    char infoLog[maxInfoLogSize];
    GLint succeedToLink = false;
    glGetProgramiv(programStageHandle, GL_LINK_STATUS, &succeedToLink);
    if (!succeedToLink) {
        glGetProgramInfoLog(programStageHandle, maxInfoLogSize, nullptr, infoLog);
        SYRINX_ERROR_FMT("fail to create program stage[name={}] because [{}]", getName(), std::string(infoLog));
        return false;
    }
    setHandle(programStageHandle);
    SYRINX_ENSURE(isCreated());
    return true;
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


bool ProgramStage::isValidParameterLocation(GLint location) const
{
    return location >= 0;
}


bool ProgramStage::isValidToCreate() const
{
    return (mType._value != ProgramStageType::UndefinedStage) && (!mSource.empty());
}

} // namespace Syrinx
