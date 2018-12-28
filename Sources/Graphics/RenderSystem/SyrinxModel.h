#pragma once
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <ResourceManager/SyrinxFileManager.h>
#include "RenderSystem/SyrinxMesh.h"

namespace Syrinx {

class Model
{
public:
    Model(FileManager *fileManager, const std::string& modelFilePath);
    ~Model();

    const std::vector<Mesh>& getMesh() const { return mMeshes; }

private:
    void loadModel(const std::string& modelFilePath);
    void processNode(aiNode *node, const aiScene *scene);
    Mesh processMesh(aiMesh *mesh, const aiScene *scene);
    std::vector<Texture> loadMaterialTextures(aiMaterial *material, aiTextureType type, std::string typeName);

    GLuint loadTexture(const std::string& textureFilePath);

private:

    FileManager *mFileManager;
    std::vector<Mesh> mMeshes;
    std::vector<Texture> mTexturesLoadedSet;
    std::string mModelFileDirectory;
};

} // namespace Syrinx
