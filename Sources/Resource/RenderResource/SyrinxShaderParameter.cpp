#include "RenderResource/SyrinxShaderParameter.h"
#include <sstream>
#include <string>
#include <algorithm>
#include <Common/SyrinxAssert.h>
#include <Container/String.h>
#include <Exception/SyrinxException.h>

namespace Syrinx {

ShaderParameter::ShaderParameter()
    : mName()
    , mType(ShaderParameterType::UNDEFINED)
    , mValue()
{
    SYRINX_ENSURE(mName.empty());
    SYRINX_ENSURE(mType._value == ShaderParameterType::UNDEFINED);
}


void ShaderParameter::setName(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    mName = name;
}


void ShaderParameter::setType(const std::string& typeString)
{
    try {
        std::string type = ToUpper(typeString);
        std::replace(std::begin(type), std::end(type), '-', '_');
        mType = ShaderParameterType::_from_string(type.c_str());
    } catch (std::exception& e) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "invalid type attribute [{}] for element <parameter>", typeString);
    }
}


void ShaderParameter::setValue(const Value& value)
{
    mValue = value;
    SYRINX_ENSURE(valid());
}


const std::string& ShaderParameter::getName() const
{
    return mName;
}


ShaderParameterType ShaderParameter::getType() const
{
    return mType;
}


const ShaderParameter::Value& ShaderParameter::getValue() const
{
    return mValue;
}


ShaderParameter::Value& ShaderParameter::getValue()
{
    return mValue;
}


bool ShaderParameter::valid() const
{
    if (mType._value == ShaderParameterType::UNDEFINED) {
        return false;
    }
    const auto typeIndex = mType._value;
    if (typeIndex <= ShaderParameterType::COLOR) {
        return typeIndex == mValue.index() + 1;
    }
    return true;
}

} // namespace Syrinx