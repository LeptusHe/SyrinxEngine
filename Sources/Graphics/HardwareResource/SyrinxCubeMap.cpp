#include "SyrinxCubeMap.h"
#include <algorithm>
#include <Exception/SyrinxException.h>

namespace Syrinx {

CubeMap::CubeMap(HardwareTexture *hardwareTexture)
    : mHardwareTexture(hardwareTexture)
    , mFaceMap()
{
    SYRINX_ENSURE(mHardwareTexture);
    SYRINX_ENSURE(!mHardwareTexture->isCreated());
    SYRINX_ENSURE(mFaceMap.empty());
}


const std::string& CubeMap::getName() const
{
    SYRINX_EXPECT(mHardwareTexture);
    return mHardwareTexture->getName();
}


HardwareResource::ResourceHandle CubeMap::getHandle() const
{
    SYRINX_EXPECT(mHardwareTexture);
    return mHardwareTexture->getHandle();
}


void CubeMap::addFaces(std::vector<Image>&& faces)
{
    std::vector<Image> cubeFaces = std::move(faces);
    validateCubeFaces(cubeFaces);
    for (size_t i = 0; i < FaceDir::_size_constant; ++ i) {
        mFaceMap.insert({static_cast<uint8_t>(i), std::move(cubeFaces[i])});
    }
    SYRINX_ENSURE(mFaceMap.size() == FaceDir::_size_constant);
}


void CubeMap::validateCubeFaces(const std::vector<Image>& cubeFaces)
{
    if (cubeFaces.size() != 6) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                   "cube map [{}] doesn't have 6 faces, it only have [{}] faces",
                                   mHardwareTexture->getName(), cubeFaces.size());
    }

    SYRINX_ASSERT(cubeFaces.size() == 6);
    int imageWidth = cubeFaces[0].getWidth();
    int imageHeight = cubeFaces[0].getHeight();
    for (const auto& image : cubeFaces) {
        if ((image.getWidth() != imageWidth) || (image.getHeight() != imageHeight)) {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                       "fail to create cube map [{{}] because face [name={}, width={}, height={}] doesn't have the same width or height with others[width={}, height={}],",
                                       mHardwareTexture->getName(),
                                       image.getName(), image.getWidth(), image.getHeight(),
                                       imageWidth, imageHeight);
        }
    }
}


void CubeMap::create()
{
    SYRINX_EXPECT(mFaceMap.size() == FaceDir::_size_constant);
    SYRINX_EXPECT(mHardwareTexture);
    SYRINX_EXPECT(!mHardwareTexture->isCreated());
    const auto& image = mFaceMap.begin()->second;
    uint32_t imageWidth = image.getWidth();
    uint32_t imageHeight = image.getHeight();
    ImageFormat imageFormat = image.getFormat();

    mHardwareTexture->setWidth(imageWidth);
    mHardwareTexture->setHeight(imageHeight);
    mHardwareTexture->setDepth(6);
    mHardwareTexture->setType(TextureType::TEXTURE_CUBEMAP);
    mHardwareTexture->setPixelFormat(PixelFormat::_from_string(imageFormat._to_string()));
    mHardwareTexture->create();
    SYRINX_ENSURE(mHardwareTexture->isCreated());

    for (const auto& [faceIndex, faceImage] : mFaceMap) {
        mHardwareTexture->write3D(faceImage.getData(), static_cast<uint32_t>(faceIndex), faceImage.getWidth(), faceImage.getHeight());
    }
}

} // namespace Syrinx
