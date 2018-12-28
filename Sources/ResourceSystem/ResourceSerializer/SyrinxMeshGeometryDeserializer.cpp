#include "ResourceSerializer/SyrinxMeshGeometryDeserializer.h"
#include <Common/SyrinxAssert.h>
#include <Exception/SyrinxException.h>
#include <Logging/SyrinxLogManager.h>

namespace Syrinx {

MeshGeometryDeserializer::MeshGeometryDeserializer()
    : Deserializer()
    , mMeshGeometry(nullptr)
    , mMeshFileHeader(MESH_FILE_HEADER)
{
    SYRINX_ENSURE(!mDataStream && !mMeshGeometry && mEndian == Endian::DEFAULT);
}


void MeshGeometryDeserializer::readCustomHeader()
{
    if (readString() != mMeshFileHeader.signature) {
        SYRINX_ERROR_FMT("the signature of mesh file header in file [{}] is invalid", mDataStream->getName());
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::DeserializationError, "the signature of mesh file header in file {} is invalid", mDataStream->getName());
    }

    if (readUInt8() != mMeshFileHeader.version) {
        SYRINX_ERROR_FMT("the version of mesh file header in file [{}] is invalid", mDataStream->getName());
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::DeserializationError, "the version of mesh file header in file [{}] is invalid", mDataStream->getName());
    }

    if (readString() != mMeshFileHeader.versionInfo) {
        SYRINX_ERROR_FMT("the version info of mesh file header in file [{}] is invalid", mDataStream->getName());
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "the version info of mesh file header in file [{}] is invalid", mDataStream->getName());
    }
}


MeshGeometry MeshGeometryDeserializer::deserialize(DataStream *dataStream, Endian endian)
{
    SYRINX_EXPECT(dataStream && dataStream->isReadable());
    SYRINX_EXPECT(isClean());
    mEndian = endian;
    mDataStream = dataStream;
    MeshGeometry meshGeometry;
    mMeshGeometry = &meshGeometry;
    deserializeFileHeader();
    deserializeData();
    clear();
    SYRINX_ENSURE(isClean());
    return meshGeometry;
}


void MeshGeometryDeserializer::deserialize(DataStream *dataStream, MeshGeometry *meshGeometry, Endian endian)
{
    SYRINX_EXPECT(dataStream && dataStream->isReadable());
    SYRINX_EXPECT(meshGeometry && meshGeometry->isClean());
    mEndian = endian;
    mDataStream = dataStream;
    mMeshGeometry = meshGeometry;
    deserializeFileHeader();
    deserializeData();
    clear();
    SYRINX_ENSURE(isClean());
}


void MeshGeometryDeserializer::deserializeData()
{
    try {
        std::string meshName = readString();
        uint32_t numVertex = readUInt32();
        Point3f *positionSet = readPositionSet(numVertex);
        Normal3f *normalSet = readNormalSet(numVertex);
        std::vector<UVChannel*> uvChannelSet;
        readUVChannelSet(numVertex, &uvChannelSet);
        uint32_t numTriangle;
        uint32_t *indexSet = readIndexSet(&numTriangle);
        MeshGeometry meshGeometry(meshName, numVertex, positionSet, normalSet, uvChannelSet, numTriangle, indexSet);
        *mMeshGeometry = std::move(meshGeometry);
    } catch (std::exception& e) {
		SYRINX_ERROR_FMT("fail to deserialize mesh file [{}] because of [{}]", mDataStream->getName(), e.what());
		SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::DeserializationError, "fail to deserialize mesh file [{}] because of [{}]", mDataStream->getName(), e.what());
    }
}


void MeshGeometryDeserializer::clear()
{
    SYRINX_EXPECT(!isClean());
    mDataStream = nullptr;
    mMeshGeometry = nullptr;
    mEndian = Endian::DEFAULT;
    SYRINX_ENSURE(isClean() && mEndian == Endian::DEFAULT);
}


bool MeshGeometryDeserializer::isClean() const
{
    return !mDataStream && !mMeshGeometry && mEndian == Endian::DEFAULT;
}


Point3f* MeshGeometryDeserializer::readPositionSet(uint32_t numVertex)
{
    bool positionSetExist = readBool();
    if (!positionSetExist)
        return nullptr;

    auto *positionSet = new Point3f[numVertex];
    readFloats(reinterpret_cast<float*>(positionSet), 3 * numVertex);
    return positionSet;
}


Normal3f* MeshGeometryDeserializer::readNormalSet(uint32_t numVertex)
{
    bool normalSetExist = readBool();
    if (!normalSetExist)
        return nullptr;

    auto *normalSet = new Normal3f[numVertex];
    readFloats(reinterpret_cast<float*>(normalSet), 3 * numVertex);
    return normalSet;
}


void MeshGeometryDeserializer::readUVChannelSet(uint32_t numVertex, std::vector<UVChannel*> *outUVChannelSet)
{
    SYRINX_EXPECT(outUVChannelSet);
    uint8_t numChannel = readUInt8();
    for (uint8_t i = 0; i < numChannel; ++i) {
        uint8_t numElement = readUInt8();
        int size = numVertex * numElement;
        auto *uvSet = new float[size];
        readFloats(uvSet, static_cast<size_t>(size));
        auto *uvChannel = new UVChannel(numElement, uvSet);
        outUVChannelSet->push_back(uvChannel);
    }
}


uint32_t* MeshGeometryDeserializer::readIndexSet(uint32_t *outNumTriangle)
{
    SYRINX_EXPECT(outNumTriangle);
    uint32_t numTriangle = readUInt32();
    uint32_t numIndex = 3 * numTriangle;
    auto *indexSet = new uint32_t[numIndex];
    readUInt32s(indexSet, numIndex);
    *outNumTriangle = numTriangle;
    return indexSet;
}

} // namespace Syrinx