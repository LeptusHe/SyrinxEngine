#pragma once
#include <string>
#include <vector>
#ifndef GLEW_STATIC
#define GLEW_STATIC
#endif //GLEW_STATIC
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <Logging/SyrinxLogManager.h>

namespace Syrinx {

struct Vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoords;

    bool operator==(const Vertex& other) const {
        return position == other.position && normal == other.normal && texCoords == other.texCoords;
    }
};

struct Texture
{
    GLuint id;
    std::string samplerName;
    std::string filePath;
};

struct BoundingBox
{
    GLuint VAO;
    std::vector<glm::vec3> vertices;
    std::vector<unsigned int> indices;

    BoundingBox()
    {
        VAO = 0;
        vertices.resize(8);
        indices.resize(36);
    }

    GLuint getVAO() const { return VAO; }
    const std::vector<glm::vec3> getVertices() const { return vertices; }
    unsigned int getIndicesSize() const { SYRINX_ENSURE(indices.size() == 36); return (unsigned int)(indices.size()); }
};

class Mesh
{
public:
    Mesh(std::vector<Vertex>& vertices, std::vector<unsigned int>& indices, std::vector<Texture>& textures);
    ~Mesh() = default;

    GLuint getVAO() const { return mVAO; }
    const std::vector<Texture>& getTexture() const { return mTextures; }
    unsigned int getIndicesSize() const { return (unsigned int)(mIndices.size()); }
    const std::vector<Vertex>& getVertices() const { return mVertices; }
    const BoundingBox& getBoundingBox() const { return mBoundingBox; }

private:
    void setupMesh();
    void computeMeshBoundingBox();
    static bool comparePositionX(const Vertex& left, const Vertex& right);
    static bool comparePositionY(const Vertex& left, const Vertex& right);
    static bool comparePositionZ(const Vertex& left, const Vertex& right);

private:
    GLuint mVAO;
    std::vector<Vertex> mVertices;
    std::vector<unsigned int> mIndices;
    std::vector<Texture> mTextures;
    BoundingBox mBoundingBox;
};

} // namespace Syrinx
