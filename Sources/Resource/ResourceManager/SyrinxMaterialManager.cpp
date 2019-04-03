#include "ResourceManager/SyrinxMaterialManager.h"
#include <pugixml.hpp>
#include <Exception/SyrinxException.h>
#include "ResourceLoader/SyrinxMaterialParser.h"

namespace Syrinx {

MaterialManager::MaterialManager(FileManager *fileManager, HardwareResourceManager *hardwareResourceManager)
    : mFileManager(fileManager)
    , mHardwareResourceManager(hardwareResourceManager)
{
    SYRINX_ENSURE(mFileManager);
    SYRINX_ENSURE(mHardwareResourceManager);
}


std::unique_ptr<Material> MaterialManager::create(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    SYRINX_EXPECT(!find(name));
    MaterialParser materialParser(mFileManager, mHardwareResourceManager, this);
    auto material = materialParser.parseMaterial(name);
    SYRINX_ASSERT(material);
    return std::unique_ptr<Material>(material);
}


FileManager* MaterialManager::getFileManager() const
{
    SYRINX_EXPECT(mFileManager);
    return mFileManager;
}


Shader* MaterialManager::createShader(const std::string& name)
{
    SYRINX_EXPECT(!name.empty());
    auto shader = new Shader(name);
    addShader(shader);
    return shader;
}


Shader* MaterialManager::findShader(const std::string& name) const
{
    SYRINX_EXPECT(!name.empty());
    auto iter = mShaderMap.find(name);
    if (iter == std::end(mShaderMap)) {
        return nullptr;
    }
    return iter->second.get();
}


void MaterialManager::addShader(Shader *shader)
{
    SYRINX_EXPECT(shader);
    SYRINX_EXPECT(!findShader(shader->getName()));
    mShaderList.push_back(shader);
    mShaderMap[shader->getName()] = std::unique_ptr<Shader>(shader);
    SYRINX_ENSURE(findShader(shader->getName()) == shader);
}

} // namespace Syrinx
