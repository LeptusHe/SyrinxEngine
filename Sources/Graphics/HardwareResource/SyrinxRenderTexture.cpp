#include "SyrinxRenderTexture.h"
#include <Common/SyrinxAssert.h>
#include "SyrinxConstantTranslator.h"

namespace Syrinx {

RenderTexture::RenderTexture()
    : mName()
    , mHardwareTextureView(nullptr)
{
    SYRINX_EXPECT(mName.empty());
    SYRINX_EXPECT(!mHardwareTextureView);
}


RenderTexture::RenderTexture(const std::string& name, const HardwareTextureView *hardwareTextureView)
    : mName(name)
    , mHardwareTextureView(hardwareTextureView)
{
    SYRINX_ENSURE(!mName.empty());
    SYRINX_ENSURE(mHardwareTextureView);
    SYRINX_ENSURE(mHardwareTextureView->isCreated());
}


RenderTexture::RenderTexture(const RenderTexture& rhs)
    : mName(rhs.mName)
    , mHardwareTextureView(rhs.mHardwareTextureView)
{
    SYRINX_ENSURE(!mName.empty());
    SYRINX_ENSURE(mHardwareTextureView);
}


RenderTexture::RenderTexture(RenderTexture&& rhs) noexcept
    : mName(std::move(rhs.mName))
    , mHardwareTextureView(rhs.mHardwareTextureView)
{
    rhs.mHardwareTextureView = nullptr;
    SYRINX_ENSURE(rhs.mName.empty());
    SYRINX_ENSURE(!rhs.mHardwareTextureView);
}


RenderTexture::operator bool() const
{
    return mHardwareTextureView != nullptr;
}


RenderTexture& RenderTexture::operator=(const RenderTexture& rhs)
{
    if (this == &rhs) {
        return *this;
    }
    mName = rhs.mName;
    mHardwareTextureView = rhs.mHardwareTextureView;
    return *this;
}


void RenderTexture::setName(const std::string& name)
{
    SYRINX_EXPECT(mName.empty());
    mName = name;
    SYRINX_ENSURE(!mName.empty());
}


void RenderTexture::setTextureView(const HardwareTextureView *textureView)
{
    SYRINX_EXPECT(!mHardwareTextureView);
    SYRINX_EXPECT(textureView);
    mHardwareTextureView = textureView;
    SYRINX_ENSURE(mHardwareTextureView);
}


HardwareResource::ResourceHandle RenderTexture::getHandle() const
{
    SYRINX_EXPECT(mHardwareTextureView);
    return mHardwareTextureView->getHandle();
}


PixelFormat RenderTexture::getPixelFormat() const
{
    return mHardwareTextureView->getPixelFormat();
}


bool RenderTexture::isDepthTexture() const
{
    auto pixelFormat = mHardwareTextureView->getPixelFormat();
    return (pixelFormat._value == PixelFormat::DEPTH24) || (pixelFormat._value == PixelFormat::DEPTH32F);
}


const std::string& RenderTexture::getName() const
{
    return mName;
}


uint32_t RenderTexture::getWidth() const
{
    SYRINX_EXPECT(mHardwareTextureView);
    return mHardwareTextureView->getWidth();
}


uint32_t RenderTexture::getHeight() const
{
    SYRINX_EXPECT(mHardwareTextureView);
    return mHardwareTextureView->getHeight();
}


uint32_t RenderTexture::getDepth() const
{
    SYRINX_EXPECT(mHardwareTextureView);
    return mHardwareTextureView->getDepth();
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
