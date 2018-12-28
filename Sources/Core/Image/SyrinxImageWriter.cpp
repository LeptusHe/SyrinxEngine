#include "Image/SyrinxImageWriter.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>
#include "Common/SyrinxAssert.h"
#include "Exception/SyrinxException.h"

namespace Syrinx {

ImageWriter::ImageWriter(std::unique_ptr<FileSystem>&& fileSystem)
    : mFileSystem(std::move(fileSystem))
{
    SYRINX_ENSURE(mFileSystem);
    SYRINX_ENSURE(!fileSystem);
}


void ImageWriter::write(const Syrinx::Image& image, const std::string& directory, const std::string& name)
{
    const std::string directoryPath = mFileSystem->weaklyCanonical(directory);
    if (!mFileSystem->directoryExist(directoryPath)) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileSystemError,
                                   "fail to save image into path [{}] because directory [{}] does not exists", name, directoryPath);
    }

    int channelNumber = Image::getChannelNumberOfImageFormat(image.getFormat());
    stbi_flip_vertically_on_write(true);
    if (!image.isHDRImage()) {
        const std::string imageName = mFileSystem->combine(directoryPath, name + ".png");
        stbi_write_png(imageName.c_str(), image.getWidth(), image.getHeight(), channelNumber, image.getData(), 0);
    } else {
        const std::string imageName = mFileSystem->combine(directoryPath, name + ".hdr");
        stbi_write_hdr(imageName.c_str(), image.getWidth(), image.getHeight(), channelNumber, reinterpret_cast<const float*>(image.getData()));
    }
}

} // namespace Syrinx
