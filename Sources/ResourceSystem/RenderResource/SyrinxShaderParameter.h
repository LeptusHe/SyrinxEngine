#pragma once
#include <string>
#include <variant>
#include <vector>
#include <better-enums/enum.h>
#include <Math/SyrinxMath.h>
#include <HardwareResource/SyrinxHardwareTexture.h>

namespace Syrinx {

BETTER_ENUM(ShaderParameterType, uint8_t, UNDEFINED, INT, FLOAT, COLOR, TEXTURE_2D, TEXTURE_3D, TEXTURE_CUBE);


struct TextureValue {
public:
    using TextureUnit = size_t;

public:
    HardwareTexture *texture = nullptr;
    TextureUnit textureUnit = 0;
    float xScale = 1.0f;
    float yScale = 1.0f;
    float xOffset = 0.0f;
    float yOffset = 0.0f;
};


class ShaderParameter {
public:
    using Value = std::variant<int, float, Color, TextureValue>;

public:
    ShaderParameter();
    ~ShaderParameter() = default;

    void setName(const std::string& name);
    void setType(const std::string& typeString);
    void setValue(const Value& value);
    const std::string& getName() const;
    ShaderParameterType getType() const;
    const ShaderParameter::Value& getValue() const;
    ShaderParameter::Value& getValue();

private:
    bool valid() const;

private:
    std::string mName;
    ShaderParameterType mType;
    Value mValue;
};

} // namespace Syrinx