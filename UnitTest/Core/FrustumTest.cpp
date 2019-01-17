#include <gmock/gmock.h>
#include <Math/SyrinxFrustum.h>

using namespace testing;
using namespace Syrinx;


class FrustumTest : public Test {
public:
    void SetUp() override
    {
        mFrustum = std::make_unique<Frustum>();
        mFrustum->setFOVy(mFOVy);
        mFrustum->setAspectRatio(mAspectRatio);
        mFrustum->setNearClipDistance(mNearClipDistance);
        mFrustum->setFarClipDistance(mFarClipDistance);
        mFrustum->setPosition(mPosition);
        mFrustum->lookAt(mLookAt);
    }

protected:
    float mFOVy = 90.0;
    float mAspectRatio = 1.0;
    float mNearClipDistance = 1.0;
    float mFarClipDistance = 100.0;
    Point3f mPosition{0.0, 0.0, 0.0};
    Point3f mLookAt{0.0, 0.0, -1.0};
    std::unique_ptr<Frustum> mFrustum;
};



TEST_F(FrustumTest, valid_parameters)
{
    ASSERT_THAT(mFrustum->getFOVy(), FloatEq(mFOVy));
    ASSERT_THAT(mFrustum->getAspectRation(), FloatEq(mAspectRatio));
    ASSERT_THAT(mFrustum->getNearClipDistance(), FloatEq(mNearClipDistance));
    ASSERT_THAT(mFrustum->getFarClipDistance(), FloatEq(mFarClipDistance));

    const Point3f& position = mFrustum->getPosition();
    ASSERT_THAT(position.x, FloatEq(mPosition.x));
    ASSERT_THAT(position.y, FloatEq(mPosition.y));
    ASSERT_THAT(position.z, FloatEq(mPosition.z));

    const Vector3f& frontDir = Normalize(mLookAt - mPosition);
    const Vector3f& frustumFrontDir = mFrustum->getFrontDir();
    ASSERT_THAT(frustumFrontDir.x, FloatEq(frontDir.x));
    ASSERT_THAT(frustumFrontDir.y, FloatEq(frontDir.y));
    ASSERT_THAT(frustumFrontDir.z, FloatEq(frontDir.z));
}


TEST_F(FrustumTest, point_near_plane)
{
    ASSERT_FALSE(mFrustum->inFrustum({0.0, 0.0, 0.0}));
    ASSERT_FALSE(mFrustum->inFrustum({0.0, 0.0, -mNearClipDistance + 1}));
    ASSERT_TRUE(mFrustum->inFrustum({0.0, 0.0, -mNearClipDistance - 1}));
}


TEST_F(FrustumTest, point_far_plane)
{
    ASSERT_TRUE(mFrustum->inFrustum({0.0, 0.0, -mFarClipDistance + 1}));
    ASSERT_FALSE(mFrustum->inFrustum({0.0, 0.0, -mFarClipDistance - 1}));
}


TEST_F(FrustumTest, center_point_in_frustum)
{
    const auto frontDir = mFrustum->getFrontDir();
    const Point3f frustumCenter = mPosition + ((mNearClipDistance + mFarClipDistance) / 2.0f) * frontDir;
    ASSERT_TRUE(mFrustum->inFrustum(frustumCenter));
}


TEST_F(FrustumTest, center_aabb_in_frustum)
{
    const auto frontDir = mFrustum->getFrontDir();
    const Point3f frustumCenter = mPosition + ((mNearClipDistance + mFarClipDistance) / 2.0f) * frontDir;
    const float extent = 1.0;
    const Point3f minimumCorner = frustumCenter - Point3f(extent / 2.0f);
    const Point3f maximumCorner = frustumCenter + Point3f(extent / 2.0f);
    AxisAlignedBox axisAlignedBox(minimumCorner, maximumCorner);

    ASSERT_TRUE(mFrustum->inFrustum(axisAlignedBox));
}