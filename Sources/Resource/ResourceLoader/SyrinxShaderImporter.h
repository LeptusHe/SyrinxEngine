#pragma once
#include <memory>
#include <pugixml.hpp>
#include <FileSystem/SyrinxFileManager.h>
#include <Program/SyrinxProgramCompiler.h>
#include <Manager/SyrinxHardwareResourceManager.h>
#include "RenderResource/SyrinxShader.h"

namespace Syrinx {

class ShaderImporter {
public:
    ShaderImporter(FileManager *fileManager,
                   HardwareResourceManager *hardwareResourceManager,
                   ProgramCompiler *compiler);
    ~ShaderImporter() = default;

    std::unique_ptr<Shader> import(const std::string& fileName);

private:
    ProgramStage* compileProgram(const std::string& fileName, const ProgramStageType& type);

private:
    FileManager *mFileManager;
    HardwareResourceManager *mHardwareResourceManager;
    ProgramCompiler *mCompiler;
};

} // namespace Syrinx