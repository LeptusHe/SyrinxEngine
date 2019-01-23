#pragma once
#include <Serializer/SyrinxDeserializer.h>
#include "ResourceSerializer/SyrinxMeshFileHeader.h"
#include "ResourceSerializer/SyrinxMeshGeometry.h"

namespace Syrinx {

class MeshGeometryDeserializer : public Deserializer {
public:
    MeshGeometryDeserializer();
    ~MeshGeometryDeserializer() override = default;

    MeshGeometry deserialize(DataStream *dataStream, Endian endian = Endian::DEFAULT) noexcept(false);
    void deserialize(DataStream *dataStream, MeshGeometry *meshGeometry, Endian endian = Endian::DEFAULT) noexcept(false);

protected:
    void readCustomHeader() noexcept(false) override;
    void deserializeData() noexcept(false) override;
    void clear() override;
    bool isClean() const override;

private:
    Point3f* readPositionSet(uint32_t numVertex);
    Normal3f* readNormalSet(uint32_t numVertex);
    std::pair<Normal3f*, Normal3f*> readTangentSetAndBitangentSet(uint32_t numVertex);
    void readUVChannelSet(uint32_t numVertex, std::vector<UVChannel*> *outUVChannelSet);
    uint32_t* readIndexSet(uint32_t *outNumTriangle);

private:
    MeshGeometry *mMeshGeometry;
    MeshFileHeader mMeshFileHeader;
};

} // namespace Syrinx
