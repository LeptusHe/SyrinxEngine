#pragma once
#include <string>
#include <memory>
#include <better-enums/enum.h>

namespace Syrinx {

BETTER_ENUM(ImageFormat, uint8_t,
            RGB8,
            RGBA8,
            RGBF,
            RGBAF);


class Image {
public:
    static size_t getSizeOfImageFormat(ImageFormat format);
    static int getChannelNumberOfImageFormat(ImageFormat format);

public:
    Image(const std::string& name, ImageFormat format, int width, int height, uint8_t *data);
    Image(ImageFormat format, int width, int height, uint8_t *data);
    ~Image();
    Image(Image&& image) noexcept;
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    const std::string& getName() const;
    uint32_t getWidth() const;
    uint32_t getHeight() const;
    ImageFormat getFormat() const;
    bool isHDRImage() const;
    template <typename T = uint8_t> const T* getData() const;

private:
    std::string mName;
    int mWidth;
    int mHeight;
    ImageFormat mFormat;
    uint8_t *mData;
};


template <typename T>
const T* Image::getData() const
{
    return reinterpret_cast<T*>(mData);
}

} // namespace Syrinx