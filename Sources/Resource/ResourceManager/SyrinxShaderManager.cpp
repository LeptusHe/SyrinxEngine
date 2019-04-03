#include "ResourceManager/SyrinxShaderManager.h"
#include "RenderResource/SyrinxShader.h"
#include "ResourceLoader/SyrinxShaderParser.h"

namespace Syrinx {

ShaderManager::ShaderManager(FileManager *fileManager, HardwareResourceManager *hardwareResourceManager)
    : mFileManager(fileManager)
    , mHardwareResourceManager(hardwareResourceManager)
{
    SYRINX_ENSURE(mFileManager);
    SYRINX_ENSURE(mHardwareResourceManager);
}


std::unique_ptr<Shader> ShaderManager::create(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    SYRINX_EXPECT(!find(name));
    ShaderParser shaderParser(mFileManager, mHardwareResourceManager);
    return shaderParser.parseShader(name);
}


FileManager* ShaderManager::getFileManager() const
{
    SYRINX_EXPECT(mFileManager);
    return mFileManager;
}

} // namespace Syrinx
