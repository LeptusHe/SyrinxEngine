#include "ResourceManager/SyrinxMeshManager.h"
#include <Common/SyrinxAssert.h>
#include <Exception/SyrinxException.h>
#include "ResourceSerializer/SyrinxMeshGeometryDeserializer.h"

namespace Syrinx {

MeshManager::MeshManager(FileManager *fileManager, HardwareResourceManager *hardwareResourceManager)
    : mFileManager(fileManager)
    , mHardwareResourceManager(hardwareResourceManager)
{
    SYRINX_ENSURE(mFileManager);
    SYRINX_ENSURE(mHardwareResourceManager);
}


std::unique_ptr<Mesh> MeshManager::create(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    SYRINX_EXPECT(!find(name));
    auto [meshExist, filePath] = mFileManager->findFile(name);
    if (!meshExist) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileNotFound, "can not find mesh file [{}]", name);
    }

    auto mesh = std::make_unique<Mesh>(name, filePath, mHardwareResourceManager);
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
        throw;
    }
    return mesh;
}

} // namespace Syrinx