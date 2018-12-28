#include "ResourceSerializer/SyrinxMeshFileHeader.h"

namespace Syrinx {

constexpr uint8_t version = 0x1;
const MeshFileHeader MESH_FILE_HEADER = MeshFileHeader("[SYRINXMESH]", version, "[SyrinxMesh_Version1.0]");


MeshFileHeader::MeshFileHeader(const std::string& signature, uint8_t version, const std::string& versionInfo)
    : signature(signature)
    , version(version)
    , versionInfo(versionInfo)
{

}

} // namespace Syrinx