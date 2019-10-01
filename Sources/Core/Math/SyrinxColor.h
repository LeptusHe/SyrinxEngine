#pragma once
#include <string>
#include <glm/glm.hpp>

namespace Syrinx {

class Color {
public:
    explicit Color(const glm::vec4& color);
    explicit Color(const float* values);
    Color(float red, float green, float blue, float alpha);
    ~Color() = default;
    explicit operator glm::vec4() const;
    operator const float*() const;
    operator float*() const;
    float& operator[](int index);
    float operator[](int index) const;
    float red() const;
    float green() const;
    float blue() const;
    float alpha() const;
    std::string toString() const;

private:
    glm::vec4 mData{1.0, 0.0, 0.0, 1.0};
};

} // namespace Syrinx