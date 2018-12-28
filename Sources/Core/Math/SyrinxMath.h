#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "Math/SyrinxColor.h"

namespace Syrinx {

using Float = float;
using Vector2f = glm::vec2;
using Vector3f = glm::vec3;
using Vector4f = glm::vec4;
using Matrix3x3 = glm::mat3;
using Matrix4x4 = glm::mat4;

using Point3f = glm::vec3;
using Position = glm::vec3;
using Normal3f = glm::vec3;


template <typename T>
inline auto GetRawValue(const T& value) -> decltype(glm::value_ptr(value))
{
    return glm::value_ptr(value);
}


inline Vector3f Normalize(const Vector3f& vector)
{
    return glm::normalize(vector);
}

} // namespace Syrinx