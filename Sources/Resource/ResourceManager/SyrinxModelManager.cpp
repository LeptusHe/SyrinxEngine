#include "ResourceManager/SyrinxModelManager.h"
#include <Exception/SyrinxException.h>
#include "ResourceLoader/SyrinxXmlParser.h"

namespace Syrinx {

ModelManager::ModelManager(FileManager *fileManager, MeshManager *meshManager, MaterialManager *materialManager)
    : mFileManager(fileManager)
    , mMeshManager(meshManager)
    , mMaterialManager(materialManager)
    , mModelMap()
    , mModelList()
{
    SYRINX_ENSURE(mFileManager);
    SYRINX_ENSURE(mMeshManager);
    SYRINX_ENSURE(mMaterialManager);
    SYRINX_ENSURE(mModelMap.empty());
    SYRINX_ENSURE(mModelList.empty());
}


Model* ModelManager::createModel(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    auto [fileExist, filePath] = mFileManager->findFile(name);
    if (!fileExist) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileNotFound, "can not find model file [{}]", name);
    }
    auto fileStream = mFileManager->openFile(filePath, FileAccessMode::READ);
    if (!fileStream) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileSystemError, "fail to open model file [{}]", filePath);
    }
    
    auto model = new Model(name);
    try {
        pugi::xml_document document;
        document.load_string(fileStream->getAsString().c_str());
        auto modelNode = getChild(document, "model");
        for (const auto& meshMaterialPairNode : modelNode) {
            if (std::string(meshMaterialPairNode.name()) != "mesh-material-pair") {
                SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                           "invalid child element [{}] in element <model> in model file [{}]",
                                           meshMaterialPairNode.name(), filePath);
            }
            std::string meshFile = getAttribute(meshMaterialPairNode, "mesh-file").as_string();
            std::string materialFile = getAttribute(meshMaterialPairNode, "material-file").as_string();

            Mesh *mesh = mMeshManager->createMesh(meshFile);
            Material *material = mMaterialManager->createOrRetrieveMaterial(materialFile);
            model->addMeshMaterialPair({mesh, material});
        }
    } catch (...) {
        delete model;
        throw;
    }
    addModel(model);
    return model;
}


Model* ModelManager::findModel(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    auto iter = mModelMap.find(name);
    if (iter == std::end(mModelMap)) {
        return nullptr;
    }
    return iter->second;
}


FileManager* ModelManager::getFileManager() const
{
    return mFileManager;
}


void ModelManager::addModel(Model *model)
{
    SYRINX_EXPECT(model);
    mModelList.push_back(std::unique_ptr<Model>(model));
    mModelMap[model->getName()] = model;
    SYRINX_ENSURE(findModel(model->getName()) == model);
}

} // namespace Syrinx