#include "RenderSystem/SyrinxMesh.h"

namespace Syrinx {

Mesh::Mesh(std::vector<Vertex>& vertices, std::vector<unsigned int>& indices, std::vector<Texture>& textures) : mVAO(0), mVertices(vertices), mIndices(indices), mTextures(textures)
{
    computeMeshBoundingBox();
    setupMesh();
}


void Mesh::setupMesh()
{
    GLuint VBO, EBO;
    glCreateBuffers(1, &VBO);
    glNamedBufferStorage(VBO, mVertices.size() * sizeof(Vertex), &mVertices[0], GL_DYNAMIC_STORAGE_BIT);

    glCreateBuffers(1, &EBO);
    glNamedBufferStorage(EBO, mIndices.size() * sizeof(unsigned int), &mIndices[0], GL_DYNAMIC_STORAGE_BIT);

    glCreateVertexArrays(1, &mVAO);
    glVertexArrayVertexBuffer(mVAO, 0, VBO, 0, sizeof(Vertex));
    glEnableVertexArrayAttrib(mVAO, 0);
    glEnableVertexArrayAttrib(mVAO, 1);
    glEnableVertexArrayAttrib(mVAO, 2);
    glVertexArrayAttribFormat(mVAO, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribFormat(mVAO, 1, 3, GL_FLOAT, GL_FALSE, (GLuint)(offsetof(Vertex, normal)));
    glVertexArrayAttribFormat(mVAO, 2, 2, GL_FLOAT, GL_FALSE, (GLuint)(offsetof(Vertex, texCoords)));
    glVertexArrayAttribBinding(mVAO, 0, 0);
    glVertexArrayAttribBinding(mVAO, 1, 0);
    glVertexArrayAttribBinding(mVAO, 2, 0);
    glVertexArrayElementBuffer(mVAO, EBO);

    glBindVertexArray(0);
}


void Mesh::computeMeshBoundingBox()
{
    std::vector<Vertex> vertices(mVertices);
    auto maxPoint = glm::vec3(0.0f);
    auto minPoint = glm::vec3(0.0f);

    std::sort(vertices.begin(), vertices.end(), comparePositionX);
    minPoint.x = vertices[0].position.x;
    maxPoint.x = vertices[vertices.size() - 1].position.x;

    std::sort(vertices.begin(), vertices.end(), comparePositionY);
    minPoint.y = vertices[0].position.y;
    maxPoint.y = vertices[vertices.size() - 1].position.y;

    std::sort(vertices.begin(), vertices.end(), comparePositionZ);
    minPoint.z = vertices[0].position.z;
    maxPoint.z = vertices[vertices.size() - 1].position.z;

    mBoundingBox.vertices[0] = glm::vec3(minPoint.x, minPoint.y, minPoint.z);
    mBoundingBox.vertices[1] = glm::vec3(minPoint.x, maxPoint.y, minPoint.z);
    mBoundingBox.vertices[2] = glm::vec3(maxPoint.x, maxPoint.y, minPoint.z);
    mBoundingBox.vertices[3] = glm::vec3(maxPoint.x, minPoint.y, minPoint.z);
    mBoundingBox.vertices[4] = glm::vec3(minPoint.x, minPoint.y, maxPoint.z);
    mBoundingBox.vertices[5] = glm::vec3(minPoint.x, maxPoint.y, maxPoint.z);
    mBoundingBox.vertices[6] = glm::vec3(maxPoint.x, maxPoint.y, maxPoint.z);
    mBoundingBox.vertices[7] = glm::vec3(maxPoint.x, minPoint.y, maxPoint.z);

    unsigned int cubeIndices[] = {
            0, 1, 2,  0, 2, 3,
            4, 6, 5,  4, 7, 6,
            4, 5, 1,  4, 1, 0,
            3, 2, 6,  3, 6, 7,
            1, 5, 6,  1, 6, 2,
            4, 0, 3,  4, 3, 7
    };
    std::copy(cubeIndices, cubeIndices + 36, mBoundingBox.indices.begin());

    GLuint VBO, EBO;
    glCreateBuffers(1, &VBO);
    glNamedBufferStorage(VBO, mBoundingBox.vertices.size() * sizeof(glm::vec3), &mBoundingBox.vertices[0], GL_DYNAMIC_STORAGE_BIT);

    glCreateBuffers(1, &EBO);
    glNamedBufferStorage(EBO, mBoundingBox.indices.size() * sizeof(unsigned int), &mBoundingBox.indices[0], GL_DYNAMIC_STORAGE_BIT);

    glCreateVertexArrays(1, &mBoundingBox.VAO);
    glVertexArrayVertexBuffer(mBoundingBox.VAO, 0, VBO, 0, sizeof(glm::vec3));
    glEnableVertexArrayAttrib(mBoundingBox.VAO, 0);
    glVertexArrayAttribFormat(mBoundingBox.VAO, 0, 3, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(mBoundingBox.VAO, 0, 0);
    glVertexArrayElementBuffer(mBoundingBox.VAO, EBO);

    glBindVertexArray(0);
}


bool Mesh::comparePositionX(const Vertex& left, const Vertex& right) {
    return left.position.x < right.position.x;
}


bool Mesh::comparePositionY(const Vertex& left, const Vertex& right) {
    return left.position.y < right.position.y;
}


bool Mesh::comparePositionZ(const Vertex& left, const Vertex& right) {
    return left.position.z < right.position.z;
}

} // namespace Syrinx