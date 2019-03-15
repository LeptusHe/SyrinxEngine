#include "SyrinxMeshExporter.h"
#include <vector>
#include <Common/SyrinxAssert.h>
#include <Math/SyrinxMath.h>
#include <Exception/SyrinxException.h>
#include <ResourceSerializer/SyrinxMeshGeometry.h>
#include <ResourceSerializer/SyrinxMeshGeometrySerializer.h>

namespace Syrinx::Tool {

MeshExporter::MeshExporter(FileManager *fileManager)
    : mFileManager(fileManager)
{
    SYRINX_ENSURE(mFileManager);
}


void MeshExporter::exportMesh(const aiMesh& mesh, const std::string& outputFileName)
{
    SYRINX_ASSERT(mesh.mPrimitiveTypes == aiPrimitiveType_TRIANGLE);

    const int numTriangle = mesh.mNumVertices;
    Point3f *positionSet = nullptr;
    if (mesh.HasPositions()) {
        positionSet = new Point3f[numTriangle];
        for (int i = 0; i < numTriangle; ++i) {
            aiVector3D pos = mesh.mVertices[i];
            positionSet[i][0] = pos.x;
            positionSet[i][1] = pos.y;
            positionSet[i][2] = pos.z;
        }
    }

    Normal3f *normalSet = nullptr;
    if (mesh.HasNormals()) {
        normalSet = new Normal3f[numTriangle];
        for (int i = 0; i < numTriangle; ++i) {
            aiVector3D normal = mesh.mNormals[i];
            normalSet[i][0] = normal.x;
            normalSet[i][1] = normal.y;
            normalSet[i][2] = normal.z;
        }
    }


    Normal3f *tangentSet = nullptr;
    Normal3f *bitangentSet = nullptr;
    if (mesh.HasTangentsAndBitangents()) {
        tangentSet = new Normal3f[numTriangle];
        bitangentSet = new Normal3f[numTriangle];
        for (int i = 0; i < numTriangle; ++ i) {
            aiVector3D tangent = mesh.mTangents[i];
            aiVector3D bitangent = mesh.mBitangents[i];
            tangentSet[i][0] = tangent.x;
            tangentSet[i][1] = tangent.y;
            tangentSet[i][2] = tangent.z;
            bitangentSet[i][0] = bitangent.x;
            bitangentSet[i][1] = bitangent.y;
            bitangentSet[i][2] = bitangent.z;
        }
    }

    std::vector<UVChannel*> uvChannelSet;
    for (unsigned int i = 0; i < mesh.GetNumUVChannels(); ++i) {
        const int numComponent = mesh.mNumUVComponents[i];
        SYRINX_ASSERT(numComponent >= 1 && numComponent <= 3);
        aiVector3D *texCoordSet = mesh.mTextureCoords[i];
        auto *uvSet = new float[numComponent * numTriangle];
        for (int triangleIndex = 0; triangleIndex < numTriangle; ++triangleIndex) {
            aiVector3D texCoord = texCoordSet[triangleIndex];
            for (int componentIndex = 0; componentIndex < numComponent; ++componentIndex) {
                uvSet[triangleIndex * numComponent + componentIndex] = texCoord[componentIndex];
            }
        }
        auto *uvChannel = new UVChannel(static_cast<uint8_t>(numComponent), uvSet);
        uvChannelSet.push_back(uvChannel);
    }

    SYRINX_ASSERT(mesh.mNumFaces > 0);
    auto *indexSet = new uint32_t[3 * mesh.mNumFaces];
    for (unsigned int i = 0; i < mesh.mNumFaces; ++i) {
        aiFace face = mesh.mFaces[i];
        for (unsigned int k = 0; k < face.mNumIndices; ++k) {
            SYRINX_ASSERT(face.mNumIndices == 3);
            indexSet[3 * i + k] = face.mIndices[k];
        }
    }

    MeshGeometry meshGeometry(mesh.mName.C_Str(), mesh.mNumVertices, positionSet, normalSet, tangentSet, bitangentSet, uvChannelSet, mesh.mNumFaces, indexSet);
    MeshGeometrySerializer meshGeometrySerializer;

    auto fileStream = mFileManager->openFile(outputFileName, FileAccessMode::WRITE);
    if (!fileStream) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileSystemError, "can not open file [{}]", outputFileName);
    }
    meshGeometrySerializer.serialize(fileStream.get(), meshGeometry);
}


} // namespace Syrinx::Tool
