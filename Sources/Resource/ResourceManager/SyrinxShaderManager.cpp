#include "ResourceManager/SyrinxShaderManager.h"
#include "RenderResource/SyrinxShader.h"
#include "ResourceLoader/SyrinxShaderImporter.h"

namespace Syrinx {

ShaderManager::ShaderManager(FileManager *fileManager, HardwareResourceManager *hardwareResourceManager)
    : mFileManager(fileManager)
    , mHardwareResourceManager(hardwareResourceManager)
    , mSearchPathList()
{
    SYRINX_ENSURE(mFileManager);
    SYRINX_ENSURE(mHardwareResourceManager);
    SYRINX_ENSURE(mSearchPathList.empty());
}


void ShaderManager::addShaderSearchPath(const std::string& path)
{
    auto fileSystem = mFileManager->getFileSystem();
    SYRINX_ASSERT(fileSystem);
    if (!fileSystem->directoryExist(path)) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                   "fail to add path [{}] into shader search path because it is not a directory",
                                   path);
    }
    mSearchPathList.push_back(path);
}


std::unique_ptr<Shader> ShaderManager::create(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    SYRINX_EXPECT(!find(name));


    ShaderImporter shaderImporter(mFileManager, mHardwareResourceManager);
    return shaderImporter.import(name, mSearchPathList);
}


FileManager* ShaderManager::getFileManager() const
{
    SYRINX_EXPECT(mFileManager);
    return mFileManager;
}

} // namespace Syrinx
