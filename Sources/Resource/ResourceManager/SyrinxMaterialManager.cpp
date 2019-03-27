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


Material* MaterialManager::createOrRetrieveMaterial(const std::string& fileName)
{
    SYRINX_EXPECT(!fileName.empty());
    auto material = findMaterial(fileName);
    if (!material) {
        material = parseMaterial(fileName);
        addMaterial(fileName, material);
    }
    return material;
}


Material* MaterialManager::parseMaterial(const std::string& filePath)
{
    MaterialParser materialParser(mFileManager, mHardwareResourceManager, this);
    return materialParser.parseMaterial(filePath);
}


Material* MaterialManager::findMaterial(const std::string& name) const
{
    SYRINX_EXPECT(!name.empty());
    auto iter = mMaterialMap.find(name);
    if (iter == std::end(mMaterialMap)) {
        return nullptr;
    }
    return iter->second.get();
}


void MaterialManager::addMaterial(const std::string& fileName, Material *material)
{
    SYRINX_EXPECT(material);
    SYRINX_EXPECT(!findMaterial(fileName));
    mMaterialList.push_back(material);
    mMaterialMap[fileName] = std::unique_ptr<Material>(material);
    SYRINX_EXPECT(findMaterial(fileName) == material);
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
