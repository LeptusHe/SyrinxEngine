#include <gmock/gmock.h>
#include <Image/SyrinxImage.h>

using namespace testing;
using namespace Syrinx;


class ImageState : public Test {
public:
    void SetUp() override
    {
        const size_t byteSize = mImageWidth * mImageHeight * Image::getSizeOfImageFormat(mFormat);
        auto data = new uint8_t[byteSize];
        mImageWithName = std::make_unique<Image>(mImageName, mFormat, mImageWidth, mImageHeight, data);
        mImageWithoutName = std::make_unique<Image>(mFormat, mImageWidth, mImageHeight, data);
        delete[] data;
    }

protected:
    std::string mImageName = "testImage";
    const int mImageWidth = 1;
    const int mImageHeight = 1;
    const ImageFormat mFormat = ImageFormat::RGBA8;
    std::unique_ptr<Image> mImageWithName;
    std::unique_ptr<Image> mImageWithoutName;
};



TEST_F(ImageState, validate_name)
{
    ASSERT_THAT(mImageWithName->getName(), Eq(mImageName));
    ASSERT_THAT(mImageWithoutName->getName(), Eq(""));
}


TEST_F(ImageState, validate_size)
{
    ASSERT_THAT(mImageWithName->getWidth(), Eq(mImageWidth));
    ASSERT_THAT(mImageWithName->getHeight(), Eq(mImageHeight));
}


TEST_F(ImageState, validata_format)
{
    ASSERT_THAT(mImageWithName->getFormat()._value, Eq(mFormat._value));
}
