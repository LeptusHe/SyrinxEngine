#include "RenderResource/SyrinxMesh.h"
#include <Exception/SyrinxException.h>
#include <Logging/SyrinxLogManager.h>

namespace Syrinx {

Mesh::Mesh(const std::string& name, const std::string& filePath, HardwareResourceManager *hardwareResourceManager)
    : RenderResource(name)
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

    mVertexInputState = new VertexInputState("[name=" + getName() + ", type=vertex input state]");


    auto positionBuffer = createVertexBufferForVertexAttribute("position", 3 * sizeof(float), getPositionSet());
    mVertexInputState->addVertexAttributeDescription({0, VertexAttributeSemantic::Position, VertexAttributeDataType::FLOAT3});
    mVertexInputState->addVertexDataDescription({positionBuffer, 0, 0, 3 * sizeof(float)});

    auto normalBuffer = createVertexBufferForVertexAttribute("normal", 3 * sizeof(float), getNormalSet());
    mVertexInputState->addVertexAttributeDescription({1, VertexAttributeSemantic::Normal, VertexAttributeDataType::FLOAT3});
    mVertexInputState->addVertexDataDescription({normalBuffer, 1, 0, 3 * sizeof(float)});

    if (mMeshGeometry->uvChannelSet.empty()) {
        SYRINX_DEBUG_FMT("mesh [{}] does not have tex coord", getName());
    } else {
        if (mMeshGeometry->uvChannelSet.size() > 1) {
            SYRINX_DEBUG_FMT("mesh [{}] has {} tex coord channel", getName(), mMeshGeometry->uvChannelSet.size());
        }
        auto texCoordBuffer = createVertexBufferForVertexAttribute("tex coord", getUVChannel(0)->numElement * sizeof(float), getUVChannel(0)->uvSet);
        mVertexInputState->addVertexAttributeDescription({2, VertexAttributeSemantic::TexCoord, VertexAttributeDataType::FLOAT2});
        mVertexInputState->addVertexDataDescription({texCoordBuffer, 2, 0, 2 * sizeof(float)});
    }
    mVertexInputState->addIndexBuffer(createIndexBuffer());
    mVertexInputState->create();

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