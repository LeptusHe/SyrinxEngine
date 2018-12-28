#include <gmock/gmock.h>
#include <HardwareResource/SyrinxVertexAttributeDescription.h>

using namespace testing;
using namespace Syrinx;


class VertexAttributeDescriptionConstructorState : public Test {
public:
    void SetUp() override
    {
        mVertexAttribute = new VertexAttributeDescription(mBindingPoint, mSemantic, mDataType);
    }

    void TearDown() override
    {
        delete mVertexAttribute;
    }

protected:
    const VertexAttributeBindingPoint mBindingPoint = 1;
    const VertexAttributeSemantic mSemantic = VertexAttributeSemantic::Position;
    const VertexAttributeDataType mDataType = VertexAttributeDataType::FLOAT3;
    VertexAttributeDescription *mVertexAttribute;
};




TEST_F(VertexAttributeDescriptionConstructorState, valid_binding_point)
{
    ASSERT_THAT(mVertexAttribute->getBindingPoint(), Eq(mBindingPoint));
}


TEST_F(VertexAttributeDescriptionConstructorState, valid_semantic)
{
    ASSERT_THAT(mVertexAttribute->getSemantic(), mSemantic);
}


TEST_F(VertexAttributeDescriptionConstructorState, valid_data_type)
{
    ASSERT_THAT(mVertexAttribute->getDataType(), mDataType);
}