#pragma once
#include <memory>
#include "FileSystem/SyrinxFileSystem.h"
#include "Image/SyrinxImage.h"

namespace Syrinx {

class ImageWriter {
public:
    explicit ImageWriter(std::unique_ptr<FileSystem>&& fileSystem = std::make_unique<FileSystem>());
    virtual ~ImageWriter() = default;

    virtual void write(const Image& image, const std::string& directory, const std::string& name);

private:
    std::unique_ptr<FileSystem> mFileSystem;
};

} // namespace Syrinx
