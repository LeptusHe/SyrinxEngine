#include "SyrinxHardwareTextureView.h"
#include <Logging/SyrinxLogManager.h>
#include "SyrinxConstantTranslator.h"

namespace Syrinx {

HardwareTextureView::HardwareTextureView(const std::string& name, HardwareTexture *texture, const TextureViewDesc& viewDesc)
    : HardwareResource(name)
    , mHardwareTexture(texture)
    , mViewDesc(viewDesc)
{
    SYRINX_EXPECT(mHardwareTexture);
}


bool HardwareTextureView::create()
{
    SYRINX_EXPECT(!isCreated());
    SYRINX_EXPECT(isValidToCreate());
    GLuint handle = 0;
    glGenTextures(1, &handle);
    auto originTextureHandle = mHardwareTexture->getHandle();
    auto textureType = ConstantTranslator::getTextureType(mViewDesc.type);
    auto format = ConstantTranslator::getPixelFormat(mHardwareTexture->getPixelFormat());
    glTextureView(handle, textureType, originTextureHandle, format, mViewDesc.baseLevel, mViewDesc.levelCount, mViewDesc.baseLayer, mViewDesc.layerCount);
    setHandle(handle);
    SYRINX_ENSURE(getHandle());
    SYRINX_ENSURE(isCreated());
    return true;
}


PixelFormat HardwareTextureView::getPixelFormat() const
{
    SYRINX_EXPECT(isCreated());
    return mHardwareTexture->getPixelFormat();
}


uint32_t HardwareTextureView::getWidth() const
{
    SYRINX_EXPECT(isCreated());
    return mHardwareTexture->getWidth();
}


uint32_t HardwareTextureView::getHeight() const
{
    SYRINX_EXPECT(isCreated());
    return mHardwareTexture->getHeight();
}


uint32_t HardwareTextureView::getDepth() const
{
    SYRINX_EXPECT(isCreated());
    return mHardwareTexture->getDepth();
}


bool HardwareTextureView::isValidToCreate() const
{
    SYRINX_EXPECT(!isCreated());
    if (mViewDesc.levelCount == 0 || mViewDesc.layerCount == 0) {
        SYRINX_DEBUG_FMT("fail to create texture view [{}] because the level count or layer count must be greater than 0", getName());
        return false;
    }

    auto maxLayer = mHardwareTexture->getDepth();
    if (mViewDesc.baseLayer + mViewDesc.layerCount > maxLayer) {
        SYRINX_DEBUG_FMT("fail to create texture view [{}] because the [base layer + layer count > max layer]", getName());
        return false;
    }

    auto maxLevel = mHardwareTexture->getMaxMipMapLevel();
    if (mViewDesc.baseLevel + mViewDesc.levelCount > maxLevel) {
        SYRINX_DEBUG_FMT("fail to create texture view [{}] because the [base level + level count > max level]", getName());
        return false;
    }
    return true;
}

} // namespace Syrinx
