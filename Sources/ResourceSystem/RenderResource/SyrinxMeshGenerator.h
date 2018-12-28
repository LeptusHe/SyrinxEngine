#pragma once
#include <memory>
#include <vector>
#include <Math/SyrinxMath.h>
#include "RenderResource/SyrinxMesh.h"

namespace Syrinx {

class MeshManager;
class HardwareResourceManager;

class MeshGenerator {
public:
    using TexCoordChannel = std::pair<uint8_t, std::vector<float>>;

public:
    MeshGenerator(HardwareResourceManager *hardwareResourceManager, MeshManager *meshManager);
    ~MeshGenerator() = default;

    MeshGenerator& addPositionSet(const std::vector<Point3f>& positionSet);
    MeshGenerator& addNormalSet(const std::vector<Normal3f>& normalSet);
    MeshGenerator& addUVChannel(uint8_t numElement, std::vector<float>& uvSet);
    MeshGenerator& addIndexSet(const std::vector<uint32_t>& indexSet);
    Mesh* build(const std::string& name);
    Mesh* generateQuadMesh(float x, float y, float width, float height);

private:
    void checkMeshState() const;
    std::unique_ptr<MeshGeometry> buildMeshGeometry() const;
    void clear();
    bool isClean() const;

private:
    std::string mMeshName;
    std::vector<Point3f> mPositionSet;
    std::vector<Normal3f> mNormalSet;
    std::vector<TexCoordChannel> mUVChannelSet;
    std::vector<uint32_t> mIndexSet;
    MeshManager *mMeshManager;
    HardwareResourceManager *mHardwareResourceManager;
};

} // namespace Syrinx
