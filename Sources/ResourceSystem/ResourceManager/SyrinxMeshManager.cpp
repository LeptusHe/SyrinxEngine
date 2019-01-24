#include "ResourceManager/SyrinxMeshManager.h"
#include <Common/SyrinxAssert.h>
#include <Exception/SyrinxException.h>
#include <ResourceSerializer/SyrinxMeshGeometryDeserializer.h>

namespace Syrinx {

MeshManager::MeshManager(FileManager *fileManager, HardwareResourceManager *hardwareResourceManager)
    : mFileManager(fileManager)
    , mHardwareResourceManager(hardwareResourceManager)
    , mMeshList()
    , mMeshMap()
{
    SYRINX_ENSURE(mFileManager);
    SYRINX_ENSURE(mHardwareResourceManager);
    SYRINX_ENSURE(mMeshList.empty());
    SYRINX_ENSURE(mMeshMap.empty());
}


Mesh* MeshManager::createMesh(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    auto [meshExist, filePath] = mFileManager->findFile(name);
    if (!meshExist) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileNotFound, "can not find mesh file [{}]", name);
    }

    auto mesh = new Mesh(name, filePath, mHardwareResourceManager);
    try {
        auto fileStream = mFileManager->openFile(filePath, FileAccessMode::READ);
        if (!fileStream) {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileSystemError, "can not open file [{}]", filePath);
        }

        auto meshGeometry = std::make_unique<MeshGeometry>();
        MeshGeometryDeserializer meshGeometryDeserializer;
        meshGeometryDeserializer.deserialize(fileStream.get(), meshGeometry.get());
        mesh->setMeshGeometry(std::move(meshGeometry));
        SYRINX_ASSERT(!meshGeometry);
    } catch (...) {
        delete mesh;
        throw;
    }
    addMesh(mesh);
    return mesh;
}


Mesh* MeshManager::findMesh(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    auto iter = mMeshMap.find(name);
    if (iter == std::end(mMeshMap)) {
        return nullptr;
    }
    return iter->second.get();
}


void MeshManager::addMesh(Mesh *mesh)
{
    SYRINX_EXPECT(mesh);
    SYRINX_EXPECT(!findMesh(mesh->getName()));
    mMeshList.push_back(mesh);
    mMeshMap[mesh->getName()] = std::unique_ptr<Mesh>(mesh);
    SYRINX_ENSURE(findMesh(mesh->getName()) == mesh);
}

} // namespace Syrinx