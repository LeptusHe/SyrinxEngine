#pragma once
#include <memory>
#include <pugixml.hpp>
#include <FileSystem/SyrinxFileManager.h>
#include <Manager/SyrinxHardwareResourceManager.h>
#include <Program/SyrinxProgramCompiler.h>
#include <Script/SyrinxLuaScript.h>
#include "RenderResource/SyrinxShader.h"

namespace Syrinx {

class ShaderImporter {
public:
    ShaderImporter(FileManager *fileManager,
                   HardwareResourceManager *hardwareResourceManager);
    ~ShaderImporter() = default;

    std::unique_ptr<Shader> import(const std::string& fileName, const std::vector<std::string>& includePathList);

private:
    ProgramCompiler::CompileOptions getCompileOption(const sol::table& table);
    ProgramStage* compileProgram(const std::string& fileName, const ProgramStageType& type, ProgramCompiler::CompileOptions&& compileOptions);

private:
    FileManager *mFileManager;
    HardwareResourceManager *mHardwareResourceManager;
};

} // namespace Syrinx