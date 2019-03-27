#pragma once
#include <memory>
#include "RenderResource/SyrinxResource.h"
#include <HardwareResource/SyrinxVertexInputState.h>
#include "ResourceSerializer/SyrinxMeshGeometry.h"
#include "ResourceManager/SyrinxHardwareResourceManager.h"

namespace Syrinx {

class Mesh : public Resource {
public:
    Mesh(const std::string& name, const std::string& filePath, HardwareResourceManager *hardwareResourceManager);
    ~Mesh() override = default;

    const std::string& getFilePath() const;
    void setMeshGeometry(std::unique_ptr<MeshGeometry>&& meshGeometry) noexcept(false);
    const MeshGeometry& getMeshGeometry() const;
    const VertexInputState& getVertexInputState() const;
    const Point3f* getPositionSet() const;
    const Normal3f* getNormalSet() const;
    const Normal3f* getTangentSet() const;
    const Normal3f* getBitangentSet() const;
    const UVChannel* getUVChannel(int index) const;
    const uint32_t* getIndexSet() const;
    size_t getNumVertex() const;
    size_t getNumTriangle() const;

private:
    void createVertexInputState() noexcept(false);
    template <typename T> HardwareVertexBuffer *createVertexBufferForVertexAttribute(const std::string& attributeName, size_t attributeDataSize, const T *data) const;
    HardwareIndexBuffer *createIndexBuffer() const;
    std::string getVertexBufferName(const std::string& attributeName) const;
    std::string getIndexBufferName() const;

private:
    std::string mFilePath;
    std::unique_ptr<MeshGeometry> mMeshGeometry;
    std::unique_ptr<VertexInputState> mVertexInputState;
    HardwareResourceManager *mHardwareResourceManager;
};


template <typename T>
HardwareVertexBuffer* Mesh::createVertexBufferForVertexAttribute(const std::string& attributeName, size_t attributeDataSize, const T *data) const
{
    SYRINX_EXPECT(data);
    return mHardwareResourceManager->createVertexBuffer(getVertexBufferName(attributeName), getNumVertex(), attributeDataSize, data);
}

} // namespace Syrinx