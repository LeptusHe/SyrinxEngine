#include "ResourceManager/SyrinxModelManager.h"
#include <Exception/SyrinxException.h>
#include "ResourceLoader/SyrinxXmlParser.h"

namespace Syrinx {

ModelManager::ModelManager(FileManager *fileManager, MeshManager *meshManager, MaterialManager *materialManager)
    : ResourceManager<Model>()
    , mFileManager(fileManager)
    , mMeshManager(meshManager)
    , mMaterialManager(materialManager)
{
    SYRINX_ENSURE(mFileManager);
    SYRINX_ENSURE(mMeshManager);
    SYRINX_ENSURE(mMaterialManager);
}


std::unique_ptr<Model> ModelManager::create(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    SYRINX_EXPECT(!find(name));
    auto [fileExist, filePath] = mFileManager->findFile(name);
    if (!fileExist) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileNotFound, "can not find model file [{}]", name);
    }
    auto fileStream = mFileManager->openFile(filePath, FileAccessMode::READ);
    if (!fileStream) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileSystemError, "fail to open model file [{}]", filePath);
    }
    
    auto model = std::make_unique<Model>(name);
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

            Mesh *mesh = mMeshManager->createOrRetrieve(meshFile);
            Material *material = mMaterialManager->createOrRetrieve(materialFile);
            model->addMeshMaterialPair({mesh, material});
        }
    } catch (...) {
        throw;
    }
    return model;
}


FileManager* ModelManager::getFileManager() const
{
    return mFileManager;
}

} // namespace Syrinx