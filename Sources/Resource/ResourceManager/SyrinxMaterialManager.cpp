#include "ResourceManager/SyrinxMaterialManager.h"
#include <pugixml.hpp>
#include <Exception/SyrinxException.h>
#include "ResourceLoader/SyrinxMaterialImporter.h"

namespace Syrinx {

MaterialManager::MaterialManager(FileManager *fileManager, HardwareResourceManager *hardwareResourceManager, ShaderManager *shaderManager)
    : mFileManager(fileManager)
    , mHardwareResourceManager(hardwareResourceManager)
    , mShaderManager(shaderManager)
{
    SYRINX_ENSURE(mFileManager);
    SYRINX_ENSURE(mHardwareResourceManager);
    SYRINX_ENSURE(mShaderManager);
}


std::unique_ptr<Material> MaterialManager::create(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    SYRINX_EXPECT(!find(name));
    MaterialImporter materialImporter(mFileManager, mHardwareResourceManager, mShaderManager);
    auto material = materialImporter.import(name);
    SYRINX_ASSERT(material);
    return material;
}


FileManager* MaterialManager::getFileManager() const
{
    SYRINX_EXPECT(mFileManager);
    return mFileManager;
}

} // namespace Syrinx
