#pragma once
#include <memory>
#include <pugixml.hpp>
#include "RenderResource/SyrinxShader.h"
#include "ResourceManager/SyrinxFileManager.h"
#include "ResourceManager/SyrinxHardwareResourceManager.h"

namespace Syrinx {

class ShaderParser {
public:
    ShaderParser(FileManager *fileManager, HardwareResourceManager *hardwareResourceManager);
    ~ShaderParser() = default;

    std::unique_ptr<Shader> parseShader(const std::string& fileName);
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
    Shader *mShader;
    ShaderPass *mShaderPass;
};

} // namespace Syrinx