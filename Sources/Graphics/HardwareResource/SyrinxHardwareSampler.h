#pragma once
#include <memory>
#include <Math/SyrinxMath.h>
#include "SyrinxGraphicsEnums.h"
#include "HardwareResource/SyrinxHardwareResource.h"

namespace Syrinx {

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
