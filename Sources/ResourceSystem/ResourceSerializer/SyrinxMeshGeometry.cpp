#include "ResourceSerializer/SyrinxMeshGeometry.h"
#include <Common/SyrinxAssert.h>

namespace {

template <typename T>
inline bool isArrayEqual(const T *lhs, const T *rhs, int length)
{
    // TODO: float number compare
    SYRINX_ENSURE(length >= 0);
    if (!lhs && !rhs) {
        return true;
    } else if ((lhs && !rhs) || (!lhs && rhs)) {
        return false;
    } else {
        SYRINX_ASSERT(lhs && rhs);
        //for (int i = 0; i < length; ++ i) {
        //    if (lhs[i] != rhs[i])
        //        return false;
        //}
    }
    return true;
}

} // anonymous namespace

namespace Syrinx {

UVChannel::UVChannel(uint8_t numElement, const float *uvSet) : numElement(numElement), uvSet(uvSet)
{
    SYRINX_ENSURE(numElement >= 2 && numElement <= 3);
    SYRINX_ENSURE(uvSet);
}


UVChannel::~UVChannel()
{
    delete[] uvSet;
}


MeshGeometry::MeshGeometry(const std::string& meshName,
                           uint32_t vertexNumber,
                           Point3f *positions,
                           Normal3f *normals,
                           const std::vector<UVChannel*>& uvChannels,
                           uint32_t triangleNumber,
                           uint32_t *indices)
    : name(meshName)
    , numVertex(vertexNumber)
    , positionSet(positions)
    , normalSet(normals)
    , uvChannelSet(uvChannels)
    , numTriangle(triangleNumber)
    , indexSet(indices)
{
    SYRINX_ENSURE(isValid());
}


MeshGeometry::MeshGeometry()
    : name()
    , numVertex(0)
    , positionSet(nullptr)
    , normalSet(nullptr)
    , uvChannelSet()
    , numTriangle(0)
    , indexSet(nullptr)
{
    SYRINX_ENSURE(numVertex == 0 && numTriangle == 0);
    SYRINX_ENSURE(name.empty() && uvChannelSet.empty());
    SYRINX_ENSURE(!positionSet && !normalSet && !indexSet);
}


MeshGeometry::MeshGeometry(MeshGeometry&& meshGeometry) noexcept
    : name(std::move(meshGeometry.name))
    , numVertex(meshGeometry.numVertex)
    , positionSet(meshGeometry.positionSet)
    , normalSet(meshGeometry.normalSet)
    , uvChannelSet(std::move(meshGeometry.uvChannelSet))
    , numTriangle(meshGeometry.numTriangle)
    , indexSet(meshGeometry.indexSet)
{
    meshGeometry.clear();
    SYRINX_ENSURE(isValid());
    SYRINX_ENSURE(meshGeometry.isClean());
}


MeshGeometry::~MeshGeometry()
{
    SYRINX_EXPECT(isValid() || isClean());
    delete[] positionSet;
    delete[] normalSet;
    for (auto& uvChannel : uvChannelSet) {
        delete uvChannel;
    }
    delete[] indexSet;
}


MeshGeometry::MeshGeometry(const MeshGeometry& meshGeometry)
    : name(meshGeometry.name)
    , numVertex(meshGeometry.numVertex)
    , positionSet(nullptr)
    , normalSet(nullptr)
    , uvChannelSet(meshGeometry.uvChannelSet.size())
    , numTriangle(meshGeometry.numTriangle)
    , indexSet(nullptr)
{
    SYRINX_EXPECT(meshGeometry.isValid());
    if (meshGeometry.positionSet) {
        positionSet = new Point3f[meshGeometry.numVertex];
        std::copy(meshGeometry.positionSet, meshGeometry.positionSet + meshGeometry.numVertex, positionSet);
    }

    if (meshGeometry.normalSet) {
        normalSet = new Normal3f[meshGeometry.numVertex];
        std::copy(meshGeometry.normalSet, meshGeometry.normalSet + meshGeometry.numVertex, normalSet);
    }

    for (unsigned int i = 0; i < meshGeometry.uvChannelSet.size(); ++ i) {
        const UVChannel *meshUVChannel = meshGeometry.uvChannelSet[i];
        SYRINX_ASSERT(meshUVChannel);

        uint8_t numElement = meshUVChannel->numElement;
        uint32_t length = meshGeometry.numVertex * numElement;
        auto *uvSet = new float[length];
        std::copy(meshUVChannel->uvSet, meshUVChannel->uvSet + length, uvSet);
        auto *channel = new UVChannel(numElement, uvSet);

        uvChannelSet[i] = channel;
    }

    if (meshGeometry.indexSet) {
        uint32_t length = 3 * meshGeometry.numTriangle;
        indexSet = new uint32_t[length];
        std::copy(meshGeometry.indexSet, meshGeometry.indexSet + length, indexSet);
    }
    SYRINX_ENSURE(meshGeometry.isValid() && isValid());
}


MeshGeometry& MeshGeometry::operator=(const MeshGeometry& meshGeometry)
{
    SYRINX_EXPECT(meshGeometry.isValid());
    MeshGeometry mesh(meshGeometry);
    swap(mesh);
    SYRINX_ENSURE(isValid() && meshGeometry.isClean());
    return *this;
}


MeshGeometry& MeshGeometry::operator=(MeshGeometry&& meshGeometry) noexcept
{
    SYRINX_EXPECT(meshGeometry.isValid());

    if (this != &meshGeometry) {
        name = meshGeometry.name;
        numVertex = meshGeometry.numVertex;
        positionSet = meshGeometry.positionSet;
        normalSet = meshGeometry.normalSet;
        uvChannelSet = std::move(meshGeometry.uvChannelSet);
        numTriangle = meshGeometry.numTriangle;
        indexSet = meshGeometry.indexSet;
        meshGeometry.clear();
    }
    SYRINX_ENSURE(isValid() && meshGeometry.isClean());
    return *this;
}


bool MeshGeometry::operator==(const MeshGeometry& rhs) const
{
    if (this == &rhs)
        return true;

    if ((name != rhs.name) || (numVertex != rhs.numVertex) || (numTriangle != rhs.numTriangle))
        return false;

    if (uvChannelSet.size() != rhs.uvChannelSet.size()) {
        return false;
    } else {
        for (size_t i = 0; i < uvChannelSet.size(); ++ i) {
            int numChannels = uvChannelSet[i]->numElement;
            if (!isArrayEqual(uvChannelSet[i]->uvSet, rhs.uvChannelSet[i]->uvSet, numVertex * numChannels))
                return false;
        }
    }

    if (!isArrayEqual(positionSet, rhs.positionSet, 3 * numVertex))
        return false;
    if (!isArrayEqual(normalSet, rhs.normalSet, 3 * numVertex))
        return false;
    if (!isArrayEqual(indexSet, rhs.indexSet, 3 * numTriangle))
        return false;
    return true;
}


void MeshGeometry::swap(MeshGeometry& rhs) noexcept
{
    name.swap(rhs.name);
    numVertex = rhs.numVertex;
    positionSet = rhs.positionSet;
    normalSet = rhs.normalSet;
    uvChannelSet.swap(rhs.uvChannelSet);
    numTriangle = rhs.numTriangle;
    indexSet = rhs.indexSet;

    rhs.numVertex = 0;
    rhs.positionSet = nullptr;
    rhs.normalSet = nullptr;
    rhs.numTriangle = 0;
    rhs.indexSet = nullptr;
}


void swap(MeshGeometry& lhs, MeshGeometry& rhs) noexcept
{
    lhs.swap(rhs);
}


bool MeshGeometry::isValid() const
{
    return (positionSet && indexSet) && (numVertex > 0 && numTriangle > 0);
}


bool MeshGeometry::isClean() const
{
    return name.empty() && uvChannelSet.empty() && (!positionSet && !normalSet && !indexSet) && (numVertex == 0 && numTriangle == 0);
}


void MeshGeometry::clear()
{
    SYRINX_EXPECT(isValid());
    name.clear();
    numVertex = 0;
    positionSet = nullptr;
    normalSet = nullptr;
    uvChannelSet.clear();
    numTriangle = 0;
    indexSet = nullptr;
    SYRINX_ENSURE(isClean());
}

} // namespace Syrinx


namespace std {

template <>
void swap<Syrinx::MeshGeometry>(Syrinx::MeshGeometry& lhs, Syrinx::MeshGeometry& rhs) noexcept
{
    lhs.swap(rhs);
}

} // namespace std