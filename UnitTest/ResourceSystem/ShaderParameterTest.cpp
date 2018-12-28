#include <gmock/gmock.h>
#include <RenderResource/SyrinxShaderParameter.h>

using namespace testing;
using namespace Syrinx;


class ShaderParameterTypeTest : public Test {
public:
    void setType(const std::string& type)
    {
        mShaderParameter.setType(type);
    }

    int getType()
    {
        return mShaderParameter.getType()._value;
    }

protected:
    ShaderParameter mShaderParameter;
};




TEST_F(ShaderParameterTypeTest, valid_int_parameter_type)
{
    setType("int");
    ASSERT_THAT(getType(), Eq(ShaderParameterType::INT));
}


TEST_F(ShaderParameterTypeTest, valid_float_parameter_type)
{
    setType("float");
    ASSERT_THAT(getType(), Eq(ShaderParameterType::FLOAT));
}


TEST_F(ShaderParameterTypeTest, valid_color_parameter_type)
{
    setType("color");
    ASSERT_THAT(getType(), Eq(ShaderParameterType::COLOR));
}


TEST_F(ShaderParameterTypeTest, valid_texture_2d_parameter_type)
{
    setType("texture-2d");
    ASSERT_THAT(getType(), Eq(ShaderParameterType::TEXTURE_2D));
}


TEST_F(ShaderParameterTypeTest, valid_texture_3d_parameter_type)
{
    setType("texture-3d");
    ASSERT_THAT(getType(), Eq(ShaderParameterType::TEXTURE_3D));
}


TEST_F(ShaderParameterTypeTest, valid_texture_cube_parameter_type)
{
    setType("texture-cube");
    ASSERT_THAT(getType(), Eq(ShaderParameterType::TEXTURE_CUBE));
}
