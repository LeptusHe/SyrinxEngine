#pragma once
#include <memory>
#include <vector>
#include <unordered_map>
#include <FileSystem/SyrinxFileManager.h>
#include <RenderResource/SyrinxModel.h>
#include <ResourceManager/SyrinxMeshManager.h>
#include <ResourceManager/SyrinxMaterialManager.h>
#include <ResourceManager/SyrinxResourceManager.h>

namespace Syrinx {

class ModelManager : public ResourceManager<Model> {
public:
    ModelManager(FileManager *fileManager, MeshManager *meshManager, MaterialManager *materialManager);
    ~ModelManager() override = default;

    std::unique_ptr<Model> create(const std::string& name) override;
    virtual FileManager* getFileManager() const;

private:
    FileManager *mFileManager;
    MeshManager *mMeshManager;
    MaterialManager *mMaterialManager;
};

} // namespace Syrinx