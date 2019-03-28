#pragma once
#include <memory>
#include <vector>
#include <unordered_map>
#include <RenderResource/SyrinxModel.h>
#include <ResourceManager/SyrinxFileManager.h>
#include <ResourceManager/SyrinxMeshManager.h>
#include <ResourceManager/SyrinxMaterialManager.h>
#include <ResourceManager/SyrinxResourceManager.h>

namespace Syrinx {

class ModelManager : public ResourceManager<Model> {
public:
    ModelManager(FileManager *fileManager, MeshManager *meshManager, MaterialManager *materialManager);
    virtual ~ModelManager() = default;

    std::unique_ptr<Model> create(const std::string& name) override;
    virtual FileManager* getFileManager() const;

private:
    FileManager *mFileManager;
    MeshManager *mMeshManager;
    MaterialManager *mMaterialManager;
};

} // namespace Syrinx