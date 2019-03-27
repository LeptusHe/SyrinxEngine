#pragma once
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>
#include <better-enums/enum.h>
#include <Image/SyrinxImage.h>
#include <HardwareResource/SyrinxHardwareTexture.h>

namespace Syrinx {

BETTER_ENUM(FaceDir, uint8_t, PositiveX, NegativeX, PositiveY, NegativeY, PositiveZ, NegativeZ);


class CubeMap {
public:
    using CubeMapFace = std::pair<uint8_t, std::string>;
    using FaceMap = std::unordered_map<uint8_t, Image>;

public:
    explicit CubeMap(HardwareTexture *hardwareTexture);
    ~CubeMap() = default;
    const std::string& getName() const;
    HardwareResource::ResourceHandle getHandle() const;
    void addFaces(std::vector<Image>&& faces) noexcept(false);
    void create();

private:
    void validateCubeFaces(const std::vector<Image>& cubeFaces);

private:
    HardwareTexture *mHardwareTexture;
    FaceMap mFaceMap;
};

} // namespace Syrinx