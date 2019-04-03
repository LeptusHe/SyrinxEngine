#pragma once
#include <pugixml.hpp>
#include "RenderResource/SyrinxShader.h"
#include "RenderResource/SyrinxMaterial.h"
#include "ResourceManager/SyrinxFileManager.h"
#include "ResourceManager/SyrinxMaterialManager.h"
#include "ResourceManager/SyrinxHardwareResourceManager.h"

namespace Syrinx {

class MaterialParser {
public:
    MaterialParser(FileManager *fileManager, HardwareResourceManager *hardwareResourceManager, ShaderManager *shaderManager);
    ~MaterialParser() = default;

    std::unique_ptr<Material> parseMaterial(const std::string& fileName);
    void parseMaterialParameterSet(const pugi::xml_node& parameterSetNode);
    void parseMaterialParameterValue(const pugi::xml_node& parameterNode, ShaderParameter& shaderParameter);
    void clear();

private:
    FileManager *mFileManager;
    HardwareResourceManager *mHardwareResourceManager;
    ShaderManager *mShaderManager;
    Material *mMaterial;
};

} // namespace Syrinx