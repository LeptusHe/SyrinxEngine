#include "ResourceSerializer/SyrinxMeshGeometrySerializer.h"
#include <Common/SyrinxAssert.h>

namespace Syrinx {

MeshGeometrySerializer::MeshGeometrySerializer()
    : Serializer()
    , mMeshGeometry(nullptr)
    , mFileHeader(MESH_FILE_HEADER)
{
    SYRINX_ENSURE(!mDataStream && !mMeshGeometry && mEndian == Endian::DEFAULT);
}


void MeshGeometrySerializer::serialize(Syrinx::DataStream *dataStream, const MeshGeometry& meshGeometry, Endian endian)
{
    SYRINX_EXPECT(dataStream && dataStream->isWriteable());
    SYRINX_EXPECT(isClean());
    mMeshGeometry = &meshGeometry;
    mEndian = endian;
    mDataStream = dataStream;
    serializeFileHeader();
    serializeData();
    clear();
    SYRINX_ENSURE(isClean());
}


void MeshGeometrySerializer::writeCustomHeader()
{
    writeString(mFileHeader.signature);
    writeUInt8(mFileHeader.version);
    writeString(mFileHeader.versionInfo);
}


void MeshGeometrySerializer::serializeData()
{
    writeString(mMeshGeometry->name);
    writeUInt32(mMeshGeometry->numVertex);
    writePositionSet();
    writeNormalSet();
    writeUVChannelSet();
    writeIndexSet();
}


void MeshGeometrySerializer::clear()
{
    SYRINX_EXPECT(mDataStream && mMeshGeometry);
    mDataStream = nullptr;
    mMeshGeometry = nullptr;
    mEndian = Endian::DEFAULT;
    SYRINX_ENSURE(!mDataStream && !mMeshGeometry && mEndian == Endian::DEFAULT);
}


bool MeshGeometrySerializer::isClean() const
{
    return !mDataStream && !mMeshGeometry && (mEndian == Endian::DEFAULT);
}


void MeshGeometrySerializer::writePositionSet()
{
    if (!mMeshGeometry->positionSet) {
        writeBool(false);
        return;
    } else {
        writeBool(true);
    }

    writeFloats(reinterpret_cast<const float*>(mMeshGeometry->positionSet), 3 * mMeshGeometry->numVertex);
}


void MeshGeometrySerializer::writeNormalSet()
{
    if (!mMeshGeometry->normalSet) {
        writeBool(false);
        return;
    } else {
        writeBool(true);
    }
    writeFloats(reinterpret_cast<const float*>(mMeshGeometry->normalSet), 3 * mMeshGeometry->numVertex);
}


void MeshGeometrySerializer::writeUVChannelSet()
{
    SYRINX_ASSERT(mMeshGeometry->uvChannelSet.size() <= std::numeric_limits<uint8_t>::max());
    writeUInt8(static_cast<uint8_t>(mMeshGeometry->uvChannelSet.size()));
    for (const auto& uvChannel : mMeshGeometry->uvChannelSet) {
        writeUInt8(uvChannel->numElement);
        writeFloats(uvChannel->uvSet, mMeshGeometry->numVertex * uvChannel->numElement);
    }
}


void MeshGeometrySerializer::writeIndexSet()
{
    writeUInt32(mMeshGeometry->numTriangle);
    writeUInt32s(mMeshGeometry->indexSet, 3 * mMeshGeometry->numTriangle);
}

} // namespace Syrinx