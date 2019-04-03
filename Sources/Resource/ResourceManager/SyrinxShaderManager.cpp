#include "ResourceManager/SyrinxShaderManager.h"
#include "RenderResource/SyrinxShader.h"

namespace Syrinx {

ShaderManager::ShaderManager(FileManager *fileManager) : mFileManager(fileManager)
{
    SYRINX_ENSURE(mFileManager);
}


std::unique_ptr<Shader> ShaderManager::create(const std::string& name)
{

}


FileManager* ShaderManager::getFileManager() const
{
    return mFileManager;
}

} // namespace Syrinx
