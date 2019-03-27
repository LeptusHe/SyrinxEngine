#include <gmock/gmock.h>
#include <Exception/SyrinxException.h>
#include <RenderResource/SyrinxShaderPass.h>

using namespace testing;
using namespace Syrinx;


class VertexAttributeSemanticTest : public Test {
public:
    void SetUp() override
    {
        mVertexAttribute.setName("vertex attribute semantic test");
    }

    void setSemantic(const std::string& semantic)
    {
        mVertexAttribute.setSemantic(semantic);
    }

    int getSemantic() const
    {
        return mVertexAttribute.getSemantic()._value;
    }

protected:
    VertexAttribute mVertexAttribute;
};


TEST_F(VertexAttributeSemanticTest, valid_position_semantic)
{
    setSemantic("position");
    ASSERT_THAT(getSemantic(), VertexAttributeSemantic::Position);
}


TEST_F(VertexAttributeSemanticTest, valid_normal_semantic)
{
    setSemantic("normal");
    ASSERT_THAT(getSemantic(), VertexAttributeSemantic::Normal);
}


TEST_F(VertexAttributeSemanticTest, valid_texture_coordinate_semantic)
{
    setSemantic("tex-coord");
    ASSERT_THAT(getSemantic(), VertexAttributeSemantic::TexCoord);
}


TEST_F(VertexAttributeSemanticTest, valid_tangent_semantic)
{
    setSemantic("tangent");
    ASSERT_THAT(getSemantic(), VertexAttributeSemantic::Tangent);
}


TEST_F(VertexAttributeSemanticTest, invalid_semantic)
{
    ASSERT_THROW(setSemantic("invalid-semantic"), InvalidParamsException);
}



class VertexAttributeDataTypeTest : public Test {
public:
    void SetUp() override
    {
        mVertexAttribute.setName("vertex attribute data type test");
    }

    void setDataType(const std::string& dataType)
    {
        mVertexAttribute.setDataType(dataType);
    }

    int getDataType() const
    {
        return mVertexAttribute.getDataType()._value;
    }

protected:
    VertexAttribute mVertexAttribute;
};


TEST_F(VertexAttributeDataTypeTest, invalid_float3_type)
{
    setDataType("float3");
    ASSERT_THAT(getDataType(), Eq(VertexAttributeDataType::FLOAT3));
}


TEST_F(VertexAttributeDataTypeTest, invalid_float2_type)
{
    setDataType("float2");
    ASSERT_THAT(getDataType(), Eq(VertexAttributeDataType::FLOAT2));
}