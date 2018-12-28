#include "SyrinxModelExporter.h"
#include <assimp/postprocess.h>
#include <Exception/SyrinxException.h>
#include <FileSystem/SyrinxFileSystem.h>
#include <Logging/SyrinxLogManager.h>
#include <Container/String.h>
#include "SyrinxMeshExporter.h"
#include "SyrinxMaterialExporter.h"

namespace Syrinx::Tool {

ModelExporter::ModelExporter(FileManager *fileManager)
	: mFileManager(fileManager)
	, mModelElement(nullptr)
{
	SYRINX_ENSURE(mFileManager);
	SYRINX_ENSURE(!mModelElement);
}


void ModelExporter::exportModel(const std::string& modelFile, const std::string& outputDirectory, const ExporterOptions& options)
{
    SYRINX_EXPECT(!outputDirectory.empty());
    auto [fileExist, filePath] = mFileManager->findFile(modelFile);
    if (!fileExist) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileNotFound, "can not find file [{}]", modelFile);
    }

    Assimp::Importer importer;
    const auto scene = importer.ReadFile(filePath, aiProcess_Triangulate | aiProcess_GenNormals);
    if (!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidState, "fail to read model [{}] because [{}]", filePath, importer.GetErrorString());
    }

    mModelFile = ToLower(modelFile);
    mModelOutputDirectory = outputDirectory;
    mOptions = options;
    mMeshOutputDirectory = FileSystem::combine(outputDirectory, "./mesh");
    mMaterialOutputDirectory = FileSystem::combine(outputDirectory, "./material");
    FileSystem::createDirectory(mModelOutputDirectory);
    FileSystem::createDirectory(mMeshOutputDirectory);
    FileSystem::createDirectory(mMaterialOutputDirectory);

    SYRINX_EXPECT(!mModelFile.empty());
    SYRINX_EXPECT(!mModelOutputDirectory.empty());
    SYRINX_EXPECT(!mMeshOutputDirectory.empty());
    SYRINX_EXPECT(!mMaterialOutputDirectory.empty());
    processScene(*scene);
    resetToDefaultState();
    SYRINX_ENSURE(mModelFile.empty());
    SYRINX_ENSURE(mModelOutputDirectory.empty());
    SYRINX_ENSURE(mMeshOutputDirectory.empty());
    SYRINX_ENSURE(mMaterialOutputDirectory.empty());
    SYRINX_ENSURE(!mModelElement);
}



void ModelExporter::processScene(const aiScene& scene)
{
    pugi::xml_document document;
    pugi::xml_node modelElement = document.append_child("model");
    mModelElement = &modelElement;

    SYRINX_EXPECT(mModelElement);
	processNode(*scene.mRootNode, scene);

    const std::string modelFileName = FileSystem::getFileName(mModelFile);
    const std::string exportedModelFileName = modelFileName.substr(0, modelFileName.find_last_of('.')) + ".smodel";
    const std::string modelFilePath = FileSystem::combine(mModelOutputDirectory, exportedModelFileName);
    document.save_file(modelFilePath.c_str(), "    ", pugi::format_indent | pugi::format_no_declaration);
    SYRINX_INFO_FMT("succeed to export model file [{}]", modelFilePath);
}


void ModelExporter::processNode(aiNode& node, const aiScene& scene)
{
	for (unsigned int i = 0; i < node.mNumMeshes; ++i) {
		aiMesh *mesh = scene.mMeshes[node.mMeshes[i]];
        processMeshAndMaterial(*mesh, scene);
	}

	for (unsigned int i = 0; i < node.mNumChildren; ++i) {
		processNode(*node.mChildren[i], scene);
	}
}


void ModelExporter::processMeshAndMaterial(aiMesh& mesh, const aiScene& scene)
{
    MeshExporter meshExporter(mFileManager);
    const std::string meshName = ToLower(mesh.mName.C_Str());

    const std::string meshFilePath = FileSystem::combine(mMeshOutputDirectory, meshName + ".smesh");
    meshExporter.exportMesh(mesh, meshFilePath);
    SYRINX_INFO_FMT("succeed to export mesh file [{}]", meshFilePath);

    MaterialExporter materialExporter;
    auto meshMaterial = scene.mMaterials[mesh.mMaterialIndex];

    std::string materialName = meshName +  "-default";
    aiString tmpMaterialName;
    if (meshMaterial->Get(AI_MATKEY_NAME, tmpMaterialName) == AI_SUCCESS) {
        materialName = tmpMaterialName.C_Str();
    }
    materialName = ToLower(materialName);
    const std::string materialFilePath = FileSystem::combine(mMaterialOutputDirectory, materialName + ".smat");
    materialExporter.exportMaterial(*meshMaterial, materialFilePath, mOptions);
    SYRINX_INFO_FMT("succeed to export material file [{}]", materialFilePath);

    pugi::xml_node meshMaterialPairElement = createMeshMaterialPairElement(meshName + ".smesh", materialName + ".smat");
    SYRINX_ASSERT(mModelElement);
}


pugi::xml_node ModelExporter::createMeshMaterialPairElement(const std::string& meshFileName, const std::string& materialFileName)
{
    pugi::xml_node modelElement = mModelElement->append_child("mesh-material-pair");
    auto meshFileAttribute = modelElement.append_attribute("mesh-file");
    meshFileAttribute.set_value(meshFileName.c_str());
    auto materialFileAttribute = modelElement.append_attribute("material-file");
    materialFileAttribute.set_value(materialFileName.c_str());
    return modelElement;
}


void ModelExporter::resetToDefaultState()
{
    mModelFile.clear();
    mModelOutputDirectory.clear();
    mOptions = ExporterOptions();
    mMeshOutputDirectory.clear();
    mMaterialOutputDirectory.clear();
    mModelElement = nullptr;
}

} // namespace Syrinx::Tool
