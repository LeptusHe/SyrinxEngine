#pragma once
#include <memory>
#include <vector>
#include <unordered_map>
#include "RenderResource/SyrinxMesh.h"
#include "ResourceManager/SyrinxFileManager.h"
#include "ResourceManager/SyrinxResourceManager.h"
#include "ResourceManager/SyrinxHardwareResourceManager.h"

namespace Syrinx {

class MeshManager : public ResourceManager<Mesh> {
public:
    explicit MeshManager(FileManager *fileManager, HardwareResourceManager *hardwareResourceManager);
    ~MeshManager() override = default;
    std::unique_ptr<Mesh> create(const std::string& name) override;

private:
    FileManager *mFileManager;
    HardwareResourceManager *mHardwareResourceManager;
};

} // namespace Syrinx