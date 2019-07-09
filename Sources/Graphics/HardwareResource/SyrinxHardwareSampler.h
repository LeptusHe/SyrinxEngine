#pragma once
#include <memory>
#include <better-enums/enum.h>
#include <Math/SyrinxMath.h>
#include "HardwareResource/SyrinxHardwareResource.h"

namespace Syrinx {

BETTER_ENUM(TextureMinFilterMethod, uint8_t,
    NEAREST,
    LINEAR,
    NEAREST_MIPMAP_NEAREST,
    LINEAR_MIPMAP_NEAREST,
    NEAREST_MIPMAP_LINEAR,
    LINEAR_MIPMAP_LINEAR
);


BETTER_ENUM(TextureMagFilterMethod, uint8_t,
    NEAREST,
    LINEAR
);


BETTER_ENUM(TextureWrapMethod, uint8_t,
    CLAMP_TO_BORDER,
    CLAMP_TO_EDGE,
    MIRROR_CLAMP_TO_EDGE,
    REPEAT,
    MIRRORED_REPEAT
);


class SamplingSetting {
public:
    SamplingSetting();
    ~SamplingSetting() = default;

    void setMinFilterMethod(TextureMinFilterMethod method);
    void setMagFilterMethod(TextureMagFilterMethod method);
    void setWrapSMethod(TextureWrapMethod method);
    void setWrapTMethod(TextureWrapMethod method);
    void setWrapRMethod(TextureWrapMethod method);
    void setBorderColor(const Color& color);
    TextureMinFilterMethod getMinFilterMethod() const;
    TextureMagFilterMethod getMagFilterMethod() const;
    TextureWrapMethod getWrapSMethod() const;
    TextureWrapMethod getWrapTMethod() const;
    TextureWrapMethod getWrapRMethod() const;
    Color getBorderColor() const;

private:
    TextureMinFilterMethod mMinFilterMethod;
    TextureMagFilterMethod mMagFilterMethod;
    TextureWrapMethod mWrapSMethod;
    TextureWrapMethod mWrapTMethod;
    TextureWrapMethod mWrapRMethod;
    Color mBorderColor;
};


class HardwareSampler : public HardwareResource {
public:
    explicit HardwareSampler(const std::string& name);
    HardwareSampler(const std::string& name, const SamplingSetting& samplingSetting);
    ~HardwareSampler() override = default;

    void setMinFilterMethod(TextureMinFilterMethod method);
    void setMagFilterMethod(TextureMagFilterMethod method);
    void setWrapSMethod(TextureWrapMethod method);
    void setWrapTMethod(TextureWrapMethod method);
    void setWrapRMethod(TextureWrapMethod method);
    void setBorderColor(const Color& color);
    TextureMinFilterMethod getMinFilterMethod() const;
    TextureMagFilterMethod getMagFilterMethod() const;
    TextureWrapMethod getWrapSMethod() const;
    TextureWrapMethod getWrapTMethod() const;
    TextureWrapMethod getWrapRMethod() const;
    Color getBorderColor() const;
    bool create() override;

private:
    bool isValidToCreate() const override;

private:
    SamplingSetting mSamplingSetting;
};

} // namespace Syrinx
