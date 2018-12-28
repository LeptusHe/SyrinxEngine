#pragma once
#include <memory>
#include <vector>
#include <unordered_map>
#include <RenderResource/SyrinxModel.h>
#include <ResourceManager/SyrinxFileManager.h>
#include <ResourceManager/SyrinxMeshManager.h>
#include <ResourceManager/SyrinxMaterialManager.h>

namespace Syrinx {

class ModelManager {
public:
    using ModelMap = std::unordered_map<std::string, Model*>;
    using ModelList = std::vector<std::unique_ptr<Model>>;

public:
    ModelManager(FileManager *fileManager, MeshManager *meshManager, MaterialManager *materialManager);
    virtual ~ModelManager() = default;

    virtual Model* createModel(const std::string& name);
    virtual Model* findModel(const std::string& name);
    virtual FileManager* getFileManager() const;

private:
    void addModel(Model *model);

private:
    FileManager *mFileManager;
    MeshManager *mMeshManager;
    MaterialManager *mMaterialManager;
    ModelMap mModelMap;
    ModelList mModelList;
};

} // namespace Syrinx