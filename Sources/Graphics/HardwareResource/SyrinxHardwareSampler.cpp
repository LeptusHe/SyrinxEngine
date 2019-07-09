#include "HardwareResource/SyrinxHardwareSampler.h"
#include <Common/SyrinxAssert.h>
#include "SyrinxConstantTranslator.h"

namespace Syrinx {

SamplingSetting::SamplingSetting()
    : mMinFilterMethod(TextureMinFilterMethod::LINEAR)
    , mMagFilterMethod(TextureMagFilterMethod::LINEAR)
    , mWrapSMethod(TextureWrapMethod::REPEAT)
    , mWrapTMethod(TextureWrapMethod::REPEAT)
    , mWrapRMethod(TextureWrapMethod::REPEAT)
    , mBorderColor(Color(0.0f, 0.0f, 0.0f, 1.0f))
{
    SYRINX_ENSURE(mMinFilterMethod._value == TextureMinFilterMethod::LINEAR);
    SYRINX_ENSURE(mMagFilterMethod._value == TextureMagFilterMethod::LINEAR);
    SYRINX_ENSURE(mWrapSMethod._value == TextureWrapMethod::REPEAT);
    SYRINX_ENSURE(mWrapTMethod._value == TextureWrapMethod::REPEAT);
    SYRINX_ENSURE(mWrapRMethod._value == TextureWrapMethod::REPEAT);
}


void SamplingSetting::setMinFilterMethod(TextureMinFilterMethod method)
{
    mMinFilterMethod = method;
}


void SamplingSetting::setMagFilterMethod(TextureMagFilterMethod method)
{
    mMagFilterMethod = method;
}


void SamplingSetting::setWrapSMethod(TextureWrapMethod method)
{
    mWrapSMethod = method;
    SYRINX_ENSURE(mWrapSMethod._value == method);
}


void SamplingSetting::setWrapTMethod(TextureWrapMethod method)
{
    mWrapTMethod = method;
    SYRINX_ENSURE(mWrapTMethod._value == method);
}


void SamplingSetting::setWrapRMethod(TextureWrapMethod method)
{
    mWrapRMethod = method;
    SYRINX_ENSURE(mWrapRMethod._value == method);
}


void SamplingSetting::setBorderColor(const Color& color)
{
    mBorderColor = color;
}


TextureMinFilterMethod SamplingSetting::getMinFilterMethod() const
{
    return mMinFilterMethod;
}


TextureMagFilterMethod SamplingSetting::getMagFilterMethod() const
{
    return mMagFilterMethod;
}


TextureWrapMethod SamplingSetting::getWrapSMethod() const
{
    return mWrapSMethod;
}


TextureWrapMethod SamplingSetting::getWrapTMethod() const
{
    return mWrapTMethod;
}


TextureWrapMethod SamplingSetting::getWrapRMethod() const
{
    return mWrapRMethod;
}


Color SamplingSetting::getBorderColor() const
{
    return mBorderColor;
}



HardwareSampler::HardwareSampler(const std::string& name)
    : HardwareResource(name)
    , mSamplingSetting()
{

}


HardwareSampler::HardwareSampler(const std::string& name, const SamplingSetting& samplingSetting)
    : HardwareResource(name)
    , mSamplingSetting(samplingSetting)
{

}


void HardwareSampler::setMinFilterMethod(TextureMinFilterMethod method)
{
    mSamplingSetting.setMinFilterMethod(method);
}


void HardwareSampler::setMagFilterMethod(TextureMagFilterMethod method)
{
    mSamplingSetting.setMagFilterMethod(method);
}


void HardwareSampler::setWrapSMethod(TextureWrapMethod method)
{
    mSamplingSetting.setWrapSMethod(method);
}


void HardwareSampler::setWrapTMethod(TextureWrapMethod method)
{
    mSamplingSetting.setWrapTMethod(method);
}


void HardwareSampler::setWrapRMethod(TextureWrapMethod method)
{
    mSamplingSetting.setWrapRMethod(method);
}


void HardwareSampler::setBorderColor(const Color& color)
{
    mSamplingSetting.setBorderColor(color);
}


TextureMinFilterMethod HardwareSampler::getMinFilterMethod() const
{
    return mSamplingSetting.getMinFilterMethod();
}


TextureMagFilterMethod HardwareSampler::getMagFilterMethod() const
{
    return mSamplingSetting.getMagFilterMethod();
}


TextureWrapMethod HardwareSampler::getWrapSMethod() const
{
    return mSamplingSetting.getWrapSMethod();
}


TextureWrapMethod HardwareSampler::getWrapTMethod() const
{
    return mSamplingSetting.getWrapTMethod();
}


TextureWrapMethod HardwareSampler::getWrapRMethod() const
{
    return mSamplingSetting.getWrapRMethod();
}


Color HardwareSampler::getBorderColor() const
{
    return mSamplingSetting.getBorderColor();
}


bool HardwareSampler::create()
{
    SYRINX_EXPECT(!isCreated());
    GLuint handle = 0;
    glCreateSamplers(1, &handle);
    glSamplerParameteri(handle, GL_TEXTURE_WRAP_S, ConstantTranslator::getTextureWrapMethod(mSamplingSetting.getWrapSMethod()));
    glSamplerParameteri(handle, GL_TEXTURE_WRAP_T, ConstantTranslator::getTextureWrapMethod(mSamplingSetting.getWrapTMethod()));
    glSamplerParameteri(handle, GL_TEXTURE_WRAP_R, ConstantTranslator::getTextureWrapMethod(mSamplingSetting.getWrapRMethod()));
    glSamplerParameteri(handle, GL_TEXTURE_MIN_FILTER, ConstantTranslator::getTextureMinFilterMethod(mSamplingSetting.getMinFilterMethod()));
    glSamplerParameteri(handle, GL_TEXTURE_MAG_FILTER, ConstantTranslator::getTextureMagFilterMethod(mSamplingSetting.getMagFilterMethod()));
    glSamplerParameterfv(handle, GL_TEXTURE_BORDER_COLOR, mSamplingSetting.getBorderColor());
    setHandle(handle);
    SYRINX_ENSURE(isCreated());
    return true;
}


bool HardwareSampler::isValidToCreate() const
{
    return true;
}

} // namespace Syrinx