#pragma once
#include <pugixml.hpp>
#include <ResourceManager/SyrinxFileManager.h>
#include "RenderResource/SyrinxShader.h"
#include "RenderResource/SyrinxMaterial.h"
#include "ResourceManager/SyrinxMaterialManager.h"
#include "ResourceManager/SyrinxHardwareResourceManager.h"

namespace Syrinx {

class MaterialParser {
public:
    MaterialParser(FileManager *fileManager, HardwareResourceManager *hardwareResourceManager, MaterialManager *materialManager);
    ~MaterialParser() = default;

    Material* parseMaterial(const std::string& fileName);
    Shader* parseShader(const std::string& fileName);
    void parseMaterialParameterSet(const pugi::xml_node& parameterSetNode);
    void parseMaterialParameterValue(const pugi::xml_node& parameterNode, ShaderParameter& shaderParameter);
    std::vector<std::unique_ptr<ShaderParameter>> parseShaderParameterSet(const pugi::xml_node& parameterSetNode);
    std::unique_ptr<ShaderParameter> parseShaderParameter(const pugi::xml_node& parameterNode);
    ShaderParameter::Value parseShaderParameterValue(const ShaderParameter& shaderParameter, const std::string& valueString);
    std::vector<std::unique_ptr<ShaderPass>> parseShaderPassSet(const pugi::xml_node& shaderNode);
    std::unique_ptr<ShaderPass> parseShaderPass(const pugi::xml_node& shaderPassNode);
    void parseVertexProgram(const pugi::xml_node& vertexProgramNode, ShaderPass *shaderPass);
    void parseFragmentProgram(const pugi::xml_node& fragmentProgramNode, ShaderPass *shaderPass);
    std::vector<VertexAttribute> parseVertexAttributeSet(const pugi::xml_node& vertexAttributeSetNode);
    std::vector<ShaderParameter*> parseProgramParameterSet(const pugi::xml_node& programParameterSetNode);
    void clear();

private:
    FileManager *mFileManager;
    HardwareResourceManager *mHardwareResourceManager;
    MaterialManager* mMaterialManager;
    Material *mMaterial;
    Shader *mShader;
    ShaderPass *mShaderPass;
};

} // namespace Syrinx