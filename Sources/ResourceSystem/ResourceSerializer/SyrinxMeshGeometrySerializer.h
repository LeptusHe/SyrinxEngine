#pragma once
#include <Serializer/SyrinxSerializer.h>
#include "ResourceSerializer/SyrinxMeshFileHeader.h"
#include "ResourceSerializer/SyrinxMeshGeometry.h"

namespace Syrinx {

class MeshGeometrySerializer : public Serializer {
public:
    MeshGeometrySerializer();
    ~MeshGeometrySerializer() override = default;

    void serialize(DataStream *dataStream, const MeshGeometry& meshGeometry, Endian endian = Endian::DEFAULT);

private:
    void writeCustomHeader() override;
    void serializeData() override;
    void clear() override;
    bool isClean() const override;
    void writePositionSet();
    void writeNormalSet();
    void writeUVChannelSet();
    void writeIndexSet();

private:
    const MeshGeometry *mMeshGeometry;
    MeshFileHeader mFileHeader;
};

} // namespace Syrinx
