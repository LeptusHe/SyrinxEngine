#include "HardwareResource/SyrinxHardwareTexture.h"
#include "HardwareResource/SyrinxConstantTranslator.h"

namespace Syrinx {

HardwareTexture::HardwareTexture(const std::string& name)
    : HardwareResource(name)
    , mWidth(0)
    , mHeight(0)
    , mDepth(0)
    , mType(TextureType::TEXTURE_2D)
    , mPixelFormat(PixelFormat::RGBA8)
    , mSampler(nullptr)
{
    SYRINX_ENSURE(mWidth == 0 && mHeight == 0 && mDepth == 0);
    SYRINX_ENSURE(mType._value == TextureType::TEXTURE_2D);
    SYRINX_ENSURE(mPixelFormat._value == PixelFormat::RGBA8);
    SYRINX_ENSURE(!mSampler);
}


HardwareTexture::~HardwareTexture()
{
    auto handle = getHandle();
    glDeleteTextures(1, &handle);
}


void HardwareTexture::setWidth(uint32_t width)
{
    SYRINX_EXPECT(width > 0);
    SYRINX_EXPECT(!isCreated());
    mWidth = width;
    SYRINX_ENSURE(mWidth == width);
}


void HardwareTexture::setHeight(uint32_t height)
{
    SYRINX_EXPECT(height > 0);
    SYRINX_EXPECT(!isCreated());
    mHeight = height;
    SYRINX_ENSURE(mHeight == height);
}


void HardwareTexture::setDepth(uint32_t depth)
{
    SYRINX_EXPECT(depth > 0);
    SYRINX_EXPECT(!isCreated());
    mDepth = depth;
    SYRINX_ENSURE(mDepth == depth);
}


void HardwareTexture::setType(TextureType type)
{
    SYRINX_EXPECT(!isCreated());
    mType = type;
    SYRINX_ENSURE(mType._value == type);
}


void HardwareTexture::setPixelFormat(PixelFormat format)
{
    SYRINX_EXPECT(!isCreated());
    mPixelFormat = format;
    SYRINX_ENSURE(mPixelFormat == format);
}


uint32_t HardwareTexture::getWidth() const
{
    return mWidth;
}


uint32_t HardwareTexture::getHeight() const
{
    return mHeight;
}


uint32_t HardwareTexture::getDepth() const
{
    return mDepth;
}


TextureType HardwareTexture::getType() const
{
    return mType;
}


PixelFormat HardwareTexture::getPixelFormat() const
{
    return mPixelFormat;
}


void HardwareTexture::setSampler(const HardwareTextureSampler *sampler)
{
    SYRINX_EXPECT(sampler);
    mSampler = sampler;
    SYRINX_ENSURE(mSampler);
    SYRINX_ENSURE(mSampler == sampler);
}


const HardwareTextureSampler* HardwareTexture::getSampler() const
{
    return mSampler;
}


void HardwareTexture::setSamplingSetting(const TextureSamplingSetting& samplingSetting)
{
    mSamplingSetting = samplingSetting;
}


const TextureSamplingSetting& HardwareTexture::getSamplingSetting() const
{
    return mSamplingSetting;
}


bool HardwareTexture::create()
{
    SYRINX_EXPECT(!isCreated());
    SYRINX_EXPECT(isValidToCreate());

    GLuint handle = 0;
    glCreateTextures(ConstantTranslator::getTextureType(mType), 1, &handle);
    glTextureParameteri(handle, GL_TEXTURE_MIN_FILTER, ConstantTranslator::getTextureMinFilterMethod(mSamplingSetting.getMinFilterMethod()));
    glTextureParameteri(handle, GL_TEXTURE_MAG_FILTER, ConstantTranslator::getTextureMagFilterMethod(mSamplingSetting.getMagFilterMethod()));
    glTextureParameteri(handle, GL_TEXTURE_WRAP_S, ConstantTranslator::getTextureWrapMethod(mSamplingSetting.getWrapSMethod()));
    glTextureParameteri(handle, GL_TEXTURE_WRAP_T, ConstantTranslator::getTextureWrapMethod(mSamplingSetting.getWrapTMethod()));
    glTextureStorage2D(handle, getMaxMipMapLevel() + 1, ConstantTranslator::getPixelFormat(mPixelFormat), mWidth, mHeight);
    setHandle(handle);

    SYRINX_ENSURE(isCreated());
    return true;
}


void HardwareTexture::generateTextureMipMap()
{
    SYRINX_EXPECT(isCreated());
    glGenerateTextureMipmap(getHandle());
}


void HardwareTexture::write(const uint8_t *source, uint32_t level, uint32_t xOffset, uint32_t yOffset, uint32_t width, uint32_t height)
{
    SYRINX_EXPECT(isCreated());
    SYRINX_EXPECT(source);
    SYRINX_EXPECT(level == 0);
    SYRINX_EXPECT(xOffset >= 0 && width > 0 && xOffset + width <= mWidth);
    SYRINX_EXPECT(yOffset >= 0 && height > 0 && yOffset + height <= mHeight);
    auto [component, componentType] = ConstantTranslator::getPixelComponentAndComponentType(mPixelFormat);
    glTextureSubImage2D(getHandle(), level, xOffset, yOffset, width, height, component, componentType, source);
}


void HardwareTexture::write3D(const uint8_t *source, uint32_t level, uint32_t xOffset, uint32_t yOffset, uint32_t zOffset, uint32_t width, uint32_t height, uint32_t depth)
{
    SYRINX_EXPECT(isCreated());
    SYRINX_EXPECT(source);
    SYRINX_EXPECT(xOffset >= 0 && width > 0 && xOffset + width <= mWidth);
    SYRINX_EXPECT(yOffset >= 0 && height > 0 && yOffset + height <= mHeight);
    SYRINX_EXPECT(zOffset >= 0 && depth > 0 && zOffset + depth <= mDepth);
    auto [component, componentType] = ConstantTranslator::getPixelComponentAndComponentType(mPixelFormat);
    glTextureSubImage3D(getHandle(), level, xOffset, yOffset, zOffset, width, height, depth, component, componentType, source);
}


int HardwareTexture::getMaxMipMapLevel() const
{
    SYRINX_EXPECT(mWidth > 0 && mHeight > 0);
    int maxSize = std::max(mWidth, mHeight);
    int maxLevel = static_cast<int>(std::floor(std::log2(maxSize)));
    SYRINX_ENSURE(maxLevel >= 0);
    return maxLevel;
}


bool HardwareTexture::isValidToCreate() const
{
    SYRINX_EXPECT((mType._value == TextureType::TEXTURE_2D) || (mType._value == TextureType::TEXTURE_CUBEMAP));
    if ((mWidth == 0) || (mHeight == 0)) {
        return false;
    }

    if (mType._value == TextureType::TEXTURE_2D) {
        return mDepth == 0;
    }
    if (mType._value == TextureType::TEXTURE_3D) {
        return mDepth != 0;
    }
    if (mType._value == TextureType::TEXTURE_CUBEMAP) {
        return mDepth == 6;
    }
    if (mType._value == TextureType::TEXTURE_2D_ARRAY) {
        return mDepth != 0;
    }
    return true;
}

} // namespace Syrinx