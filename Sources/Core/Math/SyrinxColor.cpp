#include "Math/SyrinxColor.h"
#include <glm/gtc/type_ptr.hpp>
#include "Common/SyrinxAssert.h"
#include "Container/SyrinxString.h"

namespace Syrinx {

Color::Color(const glm::vec4& color) : mData(color)
{

}


Color::Color(const float *values) : mData(glm::make_vec4(values))
{
    SYRINX_EXPECT(values);
}


Color::Color(float red, float green, float blue, float alpha) : mData(red, green, blue, alpha)
{

}


Color::operator glm::vec4() const
{
    return mData;
}


Color::operator const float*() const
{
    return glm::value_ptr(mData);
}


Color::operator float*() const
{
    return const_cast<float*>(glm::value_ptr(mData));
}


float& Color::operator[](int index)
{
    SYRINX_EXPECT(index >= 0 && index <= 3);
    return mData[index];
}


float Color::operator[](int index) const
{
    SYRINX_EXPECT(index >= 0 && index <= 3);
    return mData[index];
}


float Color::red() const
{
    return mData[0];
}


float Color::green() const
{
    return mData[1];
}


float Color::blue() const
{
    return mData[2];
}


float Color::alpha() const
{
    return mData[3];
}

std::string Color::toString() const
{
    return SYRINX_STRING_FMT("[r={}, g={}, b={}, a={}]", mData.r, mData.g, mData.b, mData.a);
}

} // namespace Syrinx
