#include "RenderResource/SyrinxRenderTexture.h"
#include <Common/SyrinxAssert.h>
#include <HardwareResource/SyrinxConstantTranslator.h>

namespace Syrinx {

RenderTexture::RenderTexture(const std::string& name, const HardwareTexture *hardwareTexture)
    : mName(name)
    , mHardwareTexture(hardwareTexture)
{
    SYRINX_ENSURE(!mName.empty());
    SYRINX_ENSURE(mHardwareTexture);
    SYRINX_ENSURE(mHardwareTexture->isCreated());
}


HardwareResource::ResourceHandle RenderTexture::getHandle() const
{
    return mHardwareTexture->getHandle();
}


PixelFormat RenderTexture::getPixelFormat() const
{
    return mHardwareTexture->getPixelFormat();
}


bool RenderTexture::isDepthTexture() const
{
    auto pixelFormat = mHardwareTexture->getPixelFormat();
    return (pixelFormat._value == PixelFormat::DEPTH24) || (pixelFormat._value == PixelFormat::DEPTH32F);
}


const std::string& RenderTexture::getName() const
{
    return mName;
}


uint32_t RenderTexture::getWidth() const
{
    SYRINX_EXPECT(mHardwareTexture);
    return mHardwareTexture->getWidth();
}


uint32_t RenderTexture::getHeight() const
{
    SYRINX_EXPECT(mHardwareTexture);
    return mHardwareTexture->getHeight();
}


uint32_t RenderTexture::getDepth() const
{
    SYRINX_EXPECT(mHardwareTexture);
    return mHardwareTexture->getDepth();
}


Image RenderTexture::getImage() const
{
    return getSubImage(0, 0, getWidth(), getHeight(), 0);
}


Image RenderTexture::getSubImage(int xOffset, int yOffset, int width, int height, int level) const
{
    SYRINX_EXPECT(level == 0);
    SYRINX_EXPECT(xOffset >= 0 && yOffset >= 0);
    SYRINX_EXPECT(width > 0 && height > 0);
    SYRINX_EXPECT(xOffset + width <= getWidth());
    SYRINX_EXPECT(yOffset + height <= getHeight());

    const auto pixelFormat = getPixelFormat();
    auto [component, componentType] = ConstantTranslator::getPixelComponentAndComponentType(pixelFormat);
    size_t sizeOfPixel = ConstantTranslator::getSizeOfPixelFormat(pixelFormat);
    int bufferSize = static_cast<int>(sizeOfPixel) * width * height;
    auto *data = new uint8_t[bufferSize];
    glGetTextureSubImage(getHandle(), level, xOffset, yOffset, 0, width, height, 1, component, componentType, bufferSize, reinterpret_cast<void*>(data));

    auto imageFormat = ImageFormat::_from_string(pixelFormat._to_string());
    Image image(imageFormat, width, height, data);
    delete[] data;

    return image;
}

} // namespace Syrinx
