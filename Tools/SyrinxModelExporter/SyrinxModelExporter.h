#pragma once
#include <vector>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <pugixml.hpp>
#include <ResourceSerializer/SyrinxMeshGeometry.h>
#include <ResourceManager/SyrinxFileManager.h>
#include "SyrinxExporterOptions.h"

namespace Syrinx::Tool {

class ModelExporter  {
public:
	using MeshGeometryList = std::vector<Syrinx::MeshGeometry*>;

public:
	explicit ModelExporter(FileManager *fileManager);
	~ModelExporter() = default;
	void exportModel(const std::string& modelFile, const std::string& outputDirectory, const ExporterOptions& options = ExporterOptions());

private:
	void processScene(const aiScene& scene);
	void processNode(aiNode& node, const aiScene& scene);
	void processMeshAndMaterial(aiMesh& node, const aiScene& scene);
	pugi::xml_node createMeshMaterialPairElement(const std::string& meshFileName, const std::string& materialFileName);
    void resetToDefaultState();

private:
	FileManager *mFileManager;
	std::string mModelFile;
	std::string mModelOutputDirectory;
	std::string mMeshOutputDirectory;
	std::string mMaterialOutputDirectory;
	ExporterOptions mOptions;
	pugi::xml_node *mModelElement;
};

} // namespace Syrinx::Tool
