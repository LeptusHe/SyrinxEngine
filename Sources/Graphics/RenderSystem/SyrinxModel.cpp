#include "RenderSystem/SyrinxModel.h"
#include <iostream>
#include <assimp/postprocess.h>
#include <stb/stb_image.h>
#include <Exception/SyrinxException.h>
#include <FileSystem/SyrinxFileSystem.h>
#include <Logging/SyrinxLogManager.h>

namespace Syrinx {

Model::Model(FileManager *fileManager, const std::string& modelFilePath)
        : mFileManager(fileManager)
        , mModelFileDirectory("")
{
    SYRINX_ASSERT(!modelFilePath.empty());
    SYRINX_ENSURE(mFileManager);
    loadModel(modelFilePath);
}


Model::~Model()
{

}


void Model::loadModel(const std::string& modelFilePath)
{
    SYRINX_ASSERT(!modelFilePath.empty());

    auto [fileExist, filePath] = mFileManager->findFile(modelFilePath);
    if (!fileExist) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileNotFound, "can not find file [{}]", modelFilePath);
    }

    Assimp::Importer importer;
    const auto scene = importer.ReadFile(filePath, aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_FlipUVs);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidState, "fail to read model [{}] because [{}]", filePath, importer.GetErrorString());
        return;
    }

    auto fileSystem = mFileManager->getFileSystem();
    mModelFileDirectory = fileSystem->getParentPath(filePath);
    SYRINX_ENSURE(!mModelFileDirectory.empty());
    processNode(scene->mRootNode, scene);
}


void Model::processNode(aiNode *node, const aiScene *scene)
{
    for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
        auto mesh = scene->mMeshes[node->mMeshes[i]];
        mMeshes.push_back(processMesh(mesh, scene));
    }

    for (unsigned int i = 0; i < node->mNumChildren; ++i) {
        processNode(node->mChildren[i], scene);
    }
}


Mesh Model::processMesh(aiMesh *mesh, const aiScene *scene)
{
    std::vector<Vertex> verticesSet;
    std::vector<unsigned int> indicesSet;
    std::vector<Texture> textureSet;

    for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
        Vertex vertex;
        vertex.position = glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
        vertex.normal = glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
        if (mesh->mTextureCoords[0]) {
            vertex.texCoords = glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);
        }
        verticesSet.push_back(vertex);
    }

    for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
        auto face = mesh->mFaces[i];
        for (unsigned int k = 0; k < face.mNumIndices; ++k) {
            indicesSet.push_back(face.mIndices[k]);
        }
    }

    if (mesh->mMaterialIndex >= 0) {
        auto material = scene->mMaterials[mesh->mMaterialIndex];
        std::vector<Texture> diffuseMaps = loadMaterialTextures(material, aiTextureType_DIFFUSE, "uTextureDiffuse");
        textureSet.insert(textureSet.end(), diffuseMaps.begin(), diffuseMaps.end());
        std::vector<Texture> specularMaps = loadMaterialTextures(material, aiTextureType_SPECULAR, "uTextureSpecular");
        textureSet.insert(textureSet.end(), specularMaps.begin(), specularMaps.end());
    }

    return Mesh(verticesSet, indicesSet, textureSet);
}


std::vector<Texture> Model::loadMaterialTextures(aiMaterial *material, aiTextureType type, std::string typeName)
{
    auto fileSystem = mFileManager->getFileSystem();
    std::vector<Texture> texturesSet;
    for (unsigned int i = 0; i < material->GetTextureCount(type); ++i) {
        aiString str;
        material->GetTexture(type, i, &str);
        std::string filePath = fileSystem->combine(mModelFileDirectory, str.C_Str());
        bool skip = false;
        for (auto textureLoaded : mTexturesLoadedSet) {
            if (std::strcmp(textureLoaded.filePath.data(), filePath.c_str()) == 0) {
                texturesSet.push_back(textureLoaded);
                skip = true;
                break;
            }
        }

        if (!skip) {
            Texture texture;
            texture.id = loadTexture(filePath);
            texture.samplerName = typeName + std::to_string(i);
            texture.filePath = filePath;
            texturesSet.push_back(texture);
            mTexturesLoadedSet.push_back(texture);
        }
    }

    return texturesSet;
}


GLuint Model::loadTexture(const std::string& textureFilePath)
{
    SYRINX_ASSERT(!textureFilePath.empty());
    GLsizei textureWidth, textureHeight, channelCount;
    stbi_uc *textureSource = stbi_load(textureFilePath.c_str(), &textureWidth, &textureHeight, &channelCount, STBI_rgb_alpha);
    if (!textureSource) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "fail to load texture [{}] because [{}]", textureFilePath, stbi_failure_reason());
    }

    GLuint texture = 0;
    glCreateTextures(GL_TEXTURE_2D, 1, &texture);
    int maxSize = std::max(textureWidth, textureHeight);
    int maxLevel = static_cast<int>(std::ceil(std::log2(maxSize)));
    SYRINX_ENSURE(maxLevel > 0);
    glTextureStorage2D(texture, maxLevel, GL_RGBA8, textureWidth, textureHeight);
    glTextureSubImage2D(texture, 0, 0, 0, textureWidth, textureHeight, GL_RGBA, GL_UNSIGNED_BYTE, textureSource);
    glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(texture, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTextureParameteri(texture, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glGenerateTextureMipmap(texture);

    glBindTexture(GL_TEXTURE_2D, 0);
    SYRINX_ENSURE(texture != 0);
    return texture;
}

} // namespace Syrinx