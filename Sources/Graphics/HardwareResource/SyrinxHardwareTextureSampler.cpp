#include "HardwareResource/SyrinxHardwareTextureSampler.h"
#include <Common/SyrinxAssert.h>

namespace Syrinx {

TextureSamplingSetting::TextureSamplingSetting()
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


void TextureSamplingSetting::setMinFilterMethod(TextureMinFilterMethod method)
{
    mMinFilterMethod = method;
}


void TextureSamplingSetting::setMagFilterMethod(TextureMagFilterMethod method)
{
    mMagFilterMethod = method;
}


void TextureSamplingSetting::setWrapSMethod(TextureWrapMethod method)
{
    mWrapSMethod = method;
    SYRINX_ENSURE(mWrapSMethod._value == method);
}


void TextureSamplingSetting::setWrapTMethod(TextureWrapMethod method)
{
    mWrapTMethod = method;
    SYRINX_ENSURE(mWrapTMethod._value == method);
}


void TextureSamplingSetting::setWrapRMethod(TextureWrapMethod method)
{
    mWrapRMethod = method;
    SYRINX_ENSURE(mWrapRMethod._value == method);
}


void TextureSamplingSetting::setBorderColor(const Color& color)
{
    mBorderColor = color;
}


TextureMinFilterMethod TextureSamplingSetting::getMinFilterMethod() const
{
    return mMinFilterMethod;
}


TextureMagFilterMethod TextureSamplingSetting::getMagFilterMethod() const
{
    return mMagFilterMethod;
}


TextureWrapMethod TextureSamplingSetting::getWrapSMethod() const
{
    return mWrapSMethod;
}


TextureWrapMethod TextureSamplingSetting::getWrapTMethod() const
{
    return mWrapTMethod;
}


TextureWrapMethod TextureSamplingSetting::getWrapRMethod() const
{
    return mWrapRMethod;
}


Color TextureSamplingSetting::getBorderColor() const
{
    return mBorderColor;
}



HardwareTextureSampler::HardwareTextureSampler(const std::string& name)
    : HardwareResource(name)
    , mSamplingSetting()
{

}


void HardwareTextureSampler::setMinFilterMethod(TextureMinFilterMethod method)
{
    mSamplingSetting.setMinFilterMethod(method);
}


void HardwareTextureSampler::setMagFilterMethod(TextureMagFilterMethod method)
{
    mSamplingSetting.setMagFilterMethod(method);
}


void HardwareTextureSampler::setWrapSMethod(TextureWrapMethod method)
{
    mSamplingSetting.setWrapSMethod(method);
}


void HardwareTextureSampler::setWrapTMethod(TextureWrapMethod method)
{
    mSamplingSetting.setWrapTMethod(method);
}


void HardwareTextureSampler::setWrapRMethod(TextureWrapMethod method)
{
    mSamplingSetting.setWrapRMethod(method);
}


void HardwareTextureSampler::setBorderColor(const Color& color)
{
    mSamplingSetting.setBorderColor(color);
}


TextureMinFilterMethod HardwareTextureSampler::getMinFilterMethod() const
{
    return mSamplingSetting.getMinFilterMethod();
}


TextureMagFilterMethod HardwareTextureSampler::getMagFilterMethod() const
{
    return mSamplingSetting.getMagFilterMethod();
}


TextureWrapMethod HardwareTextureSampler::getWrapSMethod() const
{
    return mSamplingSetting.getWrapSMethod();
}


TextureWrapMethod HardwareTextureSampler::getWrapTMethod() const
{
    return mSamplingSetting.getWrapTMethod();
}


TextureWrapMethod HardwareTextureSampler::getWrapRMethod() const
{
    return mSamplingSetting.getWrapRMethod();
}


Color HardwareTextureSampler::getBorderColor() const
{
    return mSamplingSetting.getBorderColor();
}


bool HardwareTextureSampler::create()
{
    SYRINX_EXPECT(!isCreated());

    SYRINX_ASSERT(false && "unimplemented");
    return false;
    SYRINX_ENSURE(isCreated());
}


bool HardwareTextureSampler::isValidToCreate() const
{
    return true;
}

} // namespace Syrinx