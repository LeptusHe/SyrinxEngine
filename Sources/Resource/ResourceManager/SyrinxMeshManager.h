#pragma once
#include <memory>
#include <vector>
#include <unordered_map>
#include <FileSystem/SyrinxFileManager.h>
#include <Manager/SyrinxHardwareResourceManager.h>
#include "RenderResource/SyrinxMesh.h"
#include "ResourceManager/SyrinxResourceManager.h"

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