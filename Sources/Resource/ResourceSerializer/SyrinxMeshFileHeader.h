#pragma once
#include <cstdint>
#include <string>

namespace Syrinx {

class MeshFileHeader;
extern const MeshFileHeader MESH_FILE_HEADER;

class MeshFileHeader {
public:
    MeshFileHeader(const std::string& signature, uint8_t version, const std::string& versionInfo);

public:
    const std::string signature;
    const uint8_t version;
    const std::string versionInfo;
};

} // namespace Syrinx
