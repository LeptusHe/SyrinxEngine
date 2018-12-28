#pragma once
#include "FileSystem/SyrinxFileSystem.h"
#include "Image/SyrinxImage.h"

namespace Syrinx {

class ImageReader {
public:
    explicit ImageReader(std::unique_ptr<FileSystem>&& fileSystem = std::make_unique<FileSystem>());
    virtual ~ImageReader() = default;
    virtual Image read(const std::string& filePath, ImageFormat format) noexcept(false);

private:
    Image loadRGB8ImageFromFile(const std::string& filePath) noexcept(false);
    Image loadRGBA8ImageFromFile(const std::string& filePath) noexcept(false);
    Image loadRGBFImageFromFile(const std::string& filePath) noexcept(false);
    Image loadRGBAFImageFromFile(const std::string& filePath) noexcept(false);
    void checkImageState(const std::string& name, ImageFormat format, uint8_t *data, int actualChannels, int requiredChannels) const;

private:
    std::unique_ptr<FileSystem> mFileSystem;
};

} // namespace Syrinx
