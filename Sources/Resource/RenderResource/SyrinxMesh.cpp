#include "RenderResource/SyrinxMesh.h"
#include <Exception/SyrinxException.h>
#include <Logging/SyrinxLogManager.h>

namespace Syrinx {

Mesh::Mesh(const std::string& name, const std::string& filePath, HardwareResourceManager *hardwareResourceManager)
    : Resource(name)
    , mFilePath(filePath)
    , mMeshGeometry()
    , mVertexInputState(nullptr)
    , mHardwareResourceManager(hardwareResourceManager)
{
    SYRINX_ENSURE(!filePath.empty());
    SYRINX_ENSURE(!mMeshGeometry);
    SYRINX_ENSURE(!mVertexInputState);
    SYRINX_ENSURE(mHardwareResourceManager);
}


const std::string& Mesh::getFilePath() const
{
    SYRINX_EXPECT(!mFilePath.empty());
    return mFilePath;
}


void Mesh::setMeshGeometry(std::unique_ptr<MeshGeometry>&& meshGeometry)
{
    SYRINX_EXPECT(meshGeometry);
    mMeshGeometry = std::move(meshGeometry);
    createVertexInputState();
    SYRINX_ENSURE(mVertexInputState);
    SYRINX_ENSURE(mMeshGeometry);
}


void Mesh::createVertexInputState()
{
    SYRINX_EXPECT(mMeshGeometry);

    if (!mMeshGeometry->positionSet) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidState, "mesh {[]} does not have position", getName());
    }

    if (!mMeshGeometry->normalSet) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidState, "mesh [{}] does not have normal", getName());
    }

    if (!mMeshGeometry->tangentSet) {
        SYRINX_ASSERT(!mMeshGeometry->bitangentSet);
        SYRINX_INFO_FMT("mesh [{}] doesn't have tangent and bitangent attribute", getName());
    }

    mVertexInputState = mHardwareResourceManager->createVertexInputState("[name=" + getName() + ", type=vertex input state]");

    VertexAttributeDescription positionAttributeDesc;
    positionAttributeDesc.setSemantic(VertexAttributeSemantic::Position)
                         .setLocation(0)
                         .setBindingPoint(0)
                         .setDataOffset(0)
                         .setDataType(VertexAttributeDataType::FLOAT3);

    VertexAttributeDescription normalAttributeDesc;
    normalAttributeDesc.setSemantic(VertexAttributeSemantic::Normal)
                       .setLocation(1)
                       .setBindingPoint(1)
                       .setDataOffset(0)
                       .setDataType(VertexAttributeDataType::FLOAT3);

    VertexAttributeLayoutDesc vertexAttributeLayoutDesc;
    vertexAttributeLayoutDesc.addVertexAttributeDesc(positionAttributeDesc);
    vertexAttributeLayoutDesc.addVertexAttributeDesc(normalAttributeDesc);

    auto positionBuffer = createVertexBufferForVertexAttribute("position", 3 * sizeof(float), getPositionSet());
    auto normalBuffer = createVertexBufferForVertexAttribute("normal", 3 * sizeof(float), getNormalSet());
    mVertexInputState->setVertexBuffer(positionAttributeDesc.getBindingPoint(), positionBuffer);
    mVertexInputState->setVertexBuffer(normalAttributeDesc.getBindingPoint(), normalBuffer);

    uint8_t attributeIndex = 2;
    if (mMeshGeometry->tangentSet) {
        SYRINX_ASSERT(mMeshGeometry->bitangentSet);

        auto tangentBuffer = createVertexBufferForVertexAttribute("tangent", 3 * sizeof(float), getTangentSet());
        VertexAttributeDescription tangentAttributeDesc;
        tangentAttributeDesc.setSemantic(VertexAttributeSemantic::Tangent)
                            .setLocation(attributeIndex)
                            .setBindingPoint(attributeIndex)
                            .setDataOffset(0)
                            .setDataType(VertexAttributeDataType::FLOAT3);
        vertexAttributeLayoutDesc.addVertexAttributeDesc(tangentAttributeDesc);
        mVertexInputState->setVertexBuffer(tangentAttributeDesc.getBindingPoint(), tangentBuffer);
        attributeIndex += 1;

        auto bitangentBuffer = createVertexBufferForVertexAttribute("bitangent", 3 * sizeof(float), getBitangentSet());
        VertexAttributeDescription bitangentAttributeDesc;
        bitangentAttributeDesc.setSemantic(VertexAttributeSemantic::Bitangent)
                              .setLocation(attributeIndex)
                              .setBindingPoint(attributeIndex)
                              .setDataOffset(0)
                              .setDataType(VertexAttributeDataType::FLOAT3);
        vertexAttributeLayoutDesc.addVertexAttributeDesc(bitangentAttributeDesc);
        mVertexInputState->setVertexBuffer(bitangentAttributeDesc.getBindingPoint(), bitangentBuffer);
        attributeIndex += 1;
    }

    if (mMeshGeometry->uvChannelSet.empty()) {
        SYRINX_DEBUG_FMT("mesh [{}] does not have tex coord", getName());
    } else {
        if (mMeshGeometry->uvChannelSet.size() > 1) {
            SYRINX_DEBUG_FMT("mesh [{}] has {} tex coord channel", getName(), mMeshGeometry->uvChannelSet.size());
        }
        auto texCoordBuffer = createVertexBufferForVertexAttribute("tex coord", getUVChannel(0)->numElement * sizeof(float), getUVChannel(0)->uvSet);
        VertexAttributeDescription texCoordAttributeDesc;
        texCoordAttributeDesc.setSemantic(VertexAttributeSemantic::TexCoord)
                             .setLocation(attributeIndex)
                             .setBindingPoint(attributeIndex)
                             .setDataOffset(0)
                             .setDataType(VertexAttributeDataType::FLOAT2);
        vertexAttributeLayoutDesc.addVertexAttributeDesc(texCoordAttributeDesc);
        mVertexInputState->setVertexBuffer(texCoordAttributeDesc.getBindingPoint(), texCoordBuffer);
    }

    mVertexInputState->setVertexAttributeLayoutDesc(std::move(vertexAttributeLayoutDesc));
    mVertexInputState->setIndexBuffer(createIndexBuffer());
    mVertexInputState->setup();

    SYRINX_ENSURE(mVertexInputState);
    SYRINX_ENSURE(mVertexInputState->isCreated());
}


HardwareIndexBuffer* Mesh::createIndexBuffer() const
{
    return mHardwareResourceManager->createIndexBuffer(getIndexBufferName(), 3 * getNumTriangle(), IndexType::UINT32, getIndexSet());
}


const MeshGeometry& Mesh::getMeshGeometry() const
{
    SYRINX_EXPECT(mMeshGeometry);
    return *mMeshGeometry;
}


const VertexInputState& Mesh::getVertexInputState() const
{
    SYRINX_EXPECT(mVertexInputState);
    return *mVertexInputState;
}


const Point3f* Mesh::getPositionSet() const
{
    SYRINX_EXPECT(mMeshGeometry);
    return mMeshGeometry->positionSet;
}


const Normal3f* Mesh::getNormalSet() const
{
    SYRINX_EXPECT(mMeshGeometry);
    return mMeshGeometry->normalSet;
}


const Normal3f* Mesh::getTangentSet() const
{
    SYRINX_EXPECT(mMeshGeometry);
    return mMeshGeometry->tangentSet;
}


const Normal3f* Mesh::getBitangentSet() const
{
    SYRINX_EXPECT(mMeshGeometry);
    return mMeshGeometry->bitangentSet;
}


const UVChannel* Mesh::getUVChannel(int index) const
{
    SYRINX_EXPECT(index >= 0);
    SYRINX_EXPECT(mMeshGeometry);
    if (index < mMeshGeometry->uvChannelSet.size()) {
        return mMeshGeometry->uvChannelSet[index];
    }
    return nullptr;
}


const uint32_t* Mesh::getIndexSet() const
{
    SYRINX_EXPECT(mMeshGeometry);
    return mMeshGeometry->indexSet;
}


size_t Mesh::getNumVertex() const
{
    SYRINX_EXPECT(mMeshGeometry);
    return mMeshGeometry->numVertex;
}


size_t Mesh::getNumTriangle() const
{
    SYRINX_EXPECT(mMeshGeometry);
    return mMeshGeometry->numTriangle;
}


std::string Mesh::getVertexBufferName(const std::string& attributeName) const
{
    return "[type=vertex buffer, name=" + getName() + ", attribute=" + attributeName + "]";
}


std::string Mesh::getIndexBufferName() const
{
    return "[type=index buffer, name=" + getName() + "]";
}

} // namespace Syrinx