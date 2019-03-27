#include "RenderResource/SyrinxMeshGenerator.h"
#include <Exception/SyrinxException.h>
#include "ResourceManager/SyrinxMeshManager.h"
#include "ResourceManager/SyrinxHardwareResourceManager.h"

namespace Syrinx {

MeshGenerator::MeshGenerator(HardwareResourceManager *hardwareResourceManager, MeshManager *meshManager)
    : mMeshName()
    , mPositionSet()
    , mNormalSet()
    , mUVChannelSet()
    , mIndexSet()
    , mHardwareResourceManager(hardwareResourceManager)
    , mMeshManager(meshManager)
{
    SYRINX_ENSURE(mMeshName.empty());
    SYRINX_ENSURE(mHardwareResourceManager);
    SYRINX_ENSURE(mMeshManager);
}


MeshGenerator& MeshGenerator::addPositionSet(const std::vector<Point3f>& positionSet)
{
    mPositionSet = positionSet;
    return *this;
}


MeshGenerator& MeshGenerator::addNormalSet(const std::vector<Normal3f>& normalSet)
{
    mNormalSet = normalSet;
    return *this;
}


MeshGenerator& MeshGenerator::addUVChannel(uint8_t numElement, std::vector<float>& uvSet)
{
    mUVChannelSet.emplace_back(numElement, uvSet);
    return *this;
}


MeshGenerator& MeshGenerator::addIndexSet(const std::vector<uint32_t>& indexSet)
{
    mIndexSet = indexSet;
    return *this;
}


Mesh* MeshGenerator::build(const std::string& name)
{
    mMeshName = name;
    SYRINX_EXPECT(!mMeshName.empty());
    checkMeshState();
    std::unique_ptr<MeshGeometry> meshGeometry = buildMeshGeometry();
    auto mesh = new Mesh(name, name, mHardwareResourceManager);
    mesh->setMeshGeometry(std::move(meshGeometry));
    mMeshManager->addMesh(mesh);
    mMeshName.clear();
    SYRINX_ENSURE(mMeshName.empty());
    return mesh;
}


void MeshGenerator::checkMeshState() const
{
    if (mPositionSet.empty()) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidState, "fail to build mesh [{}] because it does not have position set", mMeshName);
    }
    if (mIndexSet.empty()) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidState, "fail to build mesh [{}] because if does not have index set", mMeshName);
    }

    auto iter = std::adjacent_find(std::begin(mUVChannelSet), std::end(mUVChannelSet), [](const TexCoordChannel& lhs, const TexCoordChannel& rhs){
        return (lhs.first == rhs.first) && (lhs.second.size() == rhs.second.size());
    });
    if (iter != std::end(mUVChannelSet)) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidState, "the element number of uv channel for mesh [{}] is not the same", mMeshName);
    }

    if (!mNormalSet.empty()) {
        if (mPositionSet.size() != mNormalSet.size()) {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidState,
                                       "the size of position set and normal set for mesh [{}] is not the same",
                                       mMeshName);
        }
    }

    if (!mUVChannelSet.empty()) {
        int numTexCoord = static_cast<int>(mUVChannelSet[0].second.size()) / mUVChannelSet[0].first;
        if (numTexCoord != mPositionSet.size()) {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidState, "ths number of tex coord is not the same as the number of position for mesh [{}]", mMeshName);
        }
    }

    if (mIndexSet.size() % 3 != 0) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidState, "invalid index size [{}] for mesh [{}]", mIndexSet.size(), mMeshName);
    }
}


std::unique_ptr<MeshGeometry> MeshGenerator::buildMeshGeometry() const
{
    auto numVertex = static_cast<uint32_t>(mPositionSet.size());

    auto meshGeometry = std::make_unique<MeshGeometry>();
    meshGeometry->name = mMeshName;
    meshGeometry->numVertex = numVertex;
    meshGeometry->numTriangle = static_cast<uint32_t>(mIndexSet.size()) / 3;

    meshGeometry->positionSet = new Position[numVertex];
    std::copy(std::begin(mPositionSet), std::end(mPositionSet), meshGeometry->positionSet);

    if (!mNormalSet.empty()) {
        meshGeometry->normalSet = new Normal3f[numVertex];
        std::copy(std::begin(mNormalSet), std::end(mNormalSet), meshGeometry->normalSet);
    }

    meshGeometry->indexSet = new uint32_t[mIndexSet.size()];
    std::copy(std::begin(mIndexSet), std::end(mIndexSet), meshGeometry->indexSet);

    for (const auto& uvChannel : mUVChannelSet) {
        auto *uvSet = new float[uvChannel.second.size()];
        std::copy(std::begin(uvChannel.second), std::end(uvChannel.second), uvSet);
        auto *texCoordChannel = new UVChannel(uvChannel.first, uvSet);
        meshGeometry->uvChannelSet.push_back(texCoordChannel);
    }
    return meshGeometry;
}


Mesh* MeshGenerator::generateQuadMesh(float x, float y, float width, float height)
{
    SYRINX_EXPECT(isClean());
    std::vector<Position> positionSet = {
            {x, y, 0},
            {x + width, y, 0},
            {x + width, y + height, 0},
            {x, y + height, 0}
    };

    std::vector<float> uvSet = {
            0.0f, 0.0f,
            1.0f, 0.0f,
            1.0f, 1.0f,
            0.0f, 1.0f
    };

    std::vector<Normal3f> normalSet = {
            {0.0, 0.0, -1.0f},
            {0.0, 0.0, -1.0f},
            {0.0, 0.0, -1.0f},
            {0.0, 0.0, -1.0f}
    };

    std::vector<uint32_t> indexSet = {
            0, 1, 2,
            0, 2, 3
    };

    addPositionSet(positionSet);
    addNormalSet(normalSet);
    addUVChannel(2, uvSet);
    addIndexSet(indexSet);
    Mesh* mesh = build(fmt::format("quad-[x={}, y={}, width={}, height={}]", x, y, width, height));
    clear();
    SYRINX_ENSURE(isClean());
    return mesh;
}


void MeshGenerator::clear()
{
    mMeshName.clear();
    mPositionSet.clear();
    mNormalSet.clear();
    mUVChannelSet.clear();
    mIndexSet.clear();
}


bool MeshGenerator::isClean() const
{
    return mMeshName.empty() &&
           mPositionSet.empty() &&
           mNormalSet.empty() &&
           mUVChannelSet.empty() &&
           mIndexSet.empty();
}

} // namespace Syrinx
