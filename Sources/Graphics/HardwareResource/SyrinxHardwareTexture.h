#pragma once
#include <cstdint>
#include <memory>
#include <better-enums/enum.h>
#include <Common/SyrinxAssert.h>
#include "HardwareResource/SyrinxPixelFormat.h"
#include "HardwareResource/SyrinxHardwareResource.h"
#include "HardwareResource/SyrinxHardwareTextureSampler.h"

namespace Syrinx {

BETTER_ENUM(TextureType, uint8_t,
    TEXTURE_2D,
    TEXTURE_3D,
    TEXTURE_CUBEMAP,
    TEXTURE_2D_ARRAY
);


class HardwareTexture : public HardwareResource {
public:
    explicit HardwareTexture(const std::string& name);
    ~HardwareTexture() override;

    void setWidth(uint32_t width);
    void setHeight(uint32_t height);
    void setDepth(uint32_t depth);
    void setType(TextureType type);
    void setPixelFormat(PixelFormat format);
    uint32_t getWidth() const;
    uint32_t getHeight() const;
    uint32_t getDepth() const;
    TextureType getType() const;
    PixelFormat getPixelFormat() const;
    template <typename T = uint8_t> void write(const T *source, uint32_t width, uint32_t height);
    template <typename T = uint8_t> void write(const T *source, uint32_t level, uint32_t width, uint32_t height);
    template <typename T = uint8_t> void write(const T *source, uint32_t xOffset, uint32_t yOffset, uint32_t width, uint32_t height);
    template <typename T = uint8_t> void write(const T *source, uint32_t level, uint32_t xOffset, uint32_t yOffset, uint32_t width, uint32_t height);
    template <typename T = uint8_t> void write3D(const T *source, uint32_t depthIndex, uint32_t width, uint32_t height);
    void setSampler(const HardwareTextureSampler *sampler);
    const HardwareTextureSampler* getSampler() const;
    void setSamplingSetting(const TextureSamplingSetting& samplingSetting);
    const TextureSamplingSetting& getSamplingSetting() const;
    bool create() override;
    int getMaxMipMapLevel() const;
    void generateTextureMipMap();

private:
    void write(const uint8_t *source, uint32_t level, uint32_t xOffset, uint32_t yOffset, uint32_t width, uint32_t height);
    void write3D(const uint8_t *source, uint32_t level, uint32_t xOffset, uint32_t yOffset, uint32_t zOffset, uint32_t width, uint32_t height, uint32_t depth);
    bool isValidToCreate() const override;

private:
    uint32_t mWidth;
    uint32_t mHeight;
    uint32_t mDepth;
    TextureType mType;
    PixelFormat mPixelFormat;
    TextureSamplingSetting mSamplingSetting;
    const HardwareTextureSampler *mSampler;
};


template <typename T>
void HardwareTexture::write(const T *source, uint32_t width, uint32_t height)
{
    write(source, 0, 0, 0, width, height);
}


template <typename T>
void HardwareTexture::write(const T *source, uint32_t xOffset, uint32_t yOffset, uint32_t width, uint32_t height)
{
    write(source, 0, xOffset, yOffset, width, height);
}


template <typename T>
void HardwareTexture::write(const T *source, uint32_t level, uint32_t width, uint32_t height)
{
    write(source, level, 0, 0, width, height);
}


template <typename T>
void HardwareTexture::write(const T *source, uint32_t level, uint32_t xOffset, uint32_t yOffset, uint32_t width, uint32_t height)
{
    write(source, level, xOffset, yOffset, width, height);
}


template <typename T>
void HardwareTexture::write3D(const T *source, uint32_t depthIndex, uint32_t width, uint32_t height)
{
    write3D(source, 0, 0, 0, depthIndex, width, height, 1);
}

} // namespace Syrinx
