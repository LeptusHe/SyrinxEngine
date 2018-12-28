#pragma once
#include <assimp/scene.h>
#include <ResourceManager/SyrinxFileManager.h>

namespace Syrinx::Tool {

class MeshExporter {
public:
    explicit MeshExporter(FileManager *fileManager);
    void exportMesh(const aiMesh& mesh, const std::string& outputFileName);

private:
    FileManager *mFileManager;
};

} // namespace Syrinx::Tool


