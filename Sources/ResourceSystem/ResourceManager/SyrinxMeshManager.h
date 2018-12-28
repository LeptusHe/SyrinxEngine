#pragma once
#include <memory>
#include <vector>
#include <unordered_map>
#include <RenderResource/SyrinxMesh.h>
#include <ResourceManager/SyrinxFileManager.h>
#include <ResourceManager/SyrinxHardwareResourceManager.h>

namespace Syrinx {

class MeshManager {
public:
    using MeshList = std::vector<Mesh*>;
    using MeshMap = std::unordered_map<std::string, std::unique_ptr<Mesh>>;

public:
    explicit MeshManager(FileManager *fileManager, HardwareResourceManager *hardwareResourceManager);
    ~MeshManager() = default;
    Mesh* createMesh(const std::string& name);
    Mesh* findMesh(const std::string& name);
    void addMesh(Mesh *mesh);

private:
    FileManager *mFileManager;
    HardwareResourceManager *mHardwareResourceManager;
    MeshList mMeshList;
    MeshMap mMeshMap;
};

} // namespace Syrinx