#include <gmock/gmock.h>
#include <HardwareResource/SyrinxVertexDataDescription.h>

using namespace testing;
using namespace Syrinx;


class VertexDataDescriptionConstructor : public Test {
public:
    void SetUp() override
    {
        auto buffer = std::make_unique<HardwareBuffer>("buffer");
        mVertexBuffer = new HardwareVertexBuffer(std::move(buffer));
        VertexDataDescription vertexDataDescription(mVertexBuffer, mBindingPoint, mOffsetOfFirstElement, mStrideBetweenElements);
    }

    void TearDown() override
    {
        delete mVertexBuffer;
        delete mVertexDataDescription;
    }


protected:
    HardwareVertexBuffer *mVertexBuffer = nullptr;
    VertexBufferBindingPoint mBindingPoint = 1;
    size_t mOffsetOfFirstElement = 3 * sizeof(float);
    size_t mStrideBetweenElements = 6 * sizeof(float);
    VertexDataDescription *mVertexDataDescription = nullptr;
};



/*
TEST_F(VertexDataDescriptionConstructor, valid_vertex_buffer)
{
    ASSERT_THAT(mVertexDataDescription->getVertexBuffer(), Eq(mVertexBuffer));
}


TEST_F(VertexDataDescriptionConstructor, valid_binding_point)
{
    ASSERT_THAT(mVertexDataDescription->getVertexBufferBindingPoint(), mBindingPoint);
}

TEST_F(VertexDataDescriptionConstructor, valid_offset)
{
    ASSERT_THAT(mVertexDataDescription->getOffsetOfFirstElement(), mOffsetOfFirstElement);
}


TEST_F(VertexDataDescriptionConstructor, valid_stride)
{
    ASSERT_THAT(mVertexDataDescription->getStrideBetweenElements(), mStrideBetweenElements);
}
*/
