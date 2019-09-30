#pragma once
#include <Script/SyrinxLuaCommon.h>
#include <FileSystem/SyrinxFileManager.h>
#include <Manager/SyrinxHardwareResourceManager.h>
#include "RenderResource/SyrinxMaterial.h"
#include "RenderResource/SyrinxShader.h"
#include "RenderResource/SyrinxShaderVars.h"
#include "ResourceManager/SyrinxShaderManager.h"

namespace Syrinx {

class MaterialImporter {
public:
    MaterialImporter(FileManager *fileManager, HardwareResourceManager *hardwareResourceManager, ShaderManager *shaderManager);
    ~MaterialImporter() = default;

    std::unique_ptr<Material> import(const std::string& fileName);

private:
    void registerVariables(sol::state *state) const;
    std::unique_ptr<ShaderVars> parseShaderVars(const sol::table& shaderParameters);
    void parseProgramParameters(const sol::table& programParameters, ProgramVars *programVars);
    void parseTexture2D(const std::string& texName, const sol::table& textureDesc, ProgramVars *programVars);
    void parseUniformBuffer(const std::string& uniformBufferName, const sol::table& uniformBufferDesc, ProgramVars *programVars);
    void parseStructInfo(const std::string& structName, const sol::table& structDesc, StructBlockInfo* structBlockInfo);
    void parseVariable(const std::string& variableName, const sol::object& value, StructMemberInfo *variableInfo);

private:
    FileManager *mFileManager;
    HardwareResourceManager *mHardwareResourceManager;
    ShaderManager *mShaderManager;
    Material *mMaterial;
};

} // namespace Syrinx