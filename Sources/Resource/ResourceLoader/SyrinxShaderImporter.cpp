#include "ResourceLoader/SyrinxShaderImporter.h"
#include <Container/SyrinxString.h>
#include <Exception/SyrinxException.h>
#include <Script/SyrinxLuaScript.h>
#include <Program/SyrinxProgramCompiler.h>
#include <ResourceManager/SyrinxShaderFileIncluder.h>

namespace Syrinx {

ShaderImporter::ShaderImporter(FileManager *fileManager, HardwareResourceManager *hardwareResourceManager)
    : mFileManager(fileManager)
    , mHardwareResourceManager(hardwareResourceManager)
{
    SYRINX_ENSURE(mFileManager);
    SYRINX_ENSURE(mHardwareResourceManager);
}


std::unique_ptr<Shader> ShaderImporter::import(const std::string& fileName, const std::vector<std::string>& includePathList)
{
    auto [fileExist, filePath] = mFileManager->findFile(fileName);
    if (!fileExist) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileNotFound, "can not find file [{}]", fileName);
    }

    sol::state luaState;
    luaState.script_file(filePath);

    sol::table shaderDesc = luaState["shader"];

    std::string vertexProgramFileName = shaderDesc["vertex_program"]["file"];
    std::string fragmentProgramFileName = shaderDesc["fragment_program"]["file"];

    const std::string& shaderName = fileName;
    auto programPipeline = mHardwareResourceManager->createProgramPipeline(shaderName);

    auto vertexProgramCompileOption = getCompileOption(shaderDesc["vertex_program"]);
    auto fragmentProgramCompileOption = getCompileOption(shaderDesc["fragment_program"]);

    auto vertexProgramStage = compileProgram(vertexProgramFileName, ProgramStageType::VertexStage, std::move(vertexProgramCompileOption), includePathList);
    auto fragmentProgramStage = compileProgram(fragmentProgramFileName, ProgramStageType::FragmentStage, std::move(fragmentProgramCompileOption), includePathList);

    programPipeline->bindProgramStage(vertexProgramStage);
    programPipeline->bindProgramStage(fragmentProgramStage);

    auto shader = std::make_unique<Shader>(shaderName);
    shader->addProgramPipeline(programPipeline);

    return shader;
}


ProgramCompiler::CompileOptions ShaderImporter::getCompileOption(const sol::table& programDesc)
{
    SYRINX_EXPECT(programDesc);
    ProgramCompiler::CompileOptions compileOptions;

    if (sol::optional<sol::table> predefinedMacros = programDesc["predefined_macros"]; predefinedMacros != sol::nullopt) {
        for (const auto& kvPair : *predefinedMacros) {
            sol::table macroDesc = kvPair.second;
            std::string macroName = macroDesc["macro"];
            std::string macroValue = macroDesc["value"];
            compileOptions.AddMacroDefinition(macroName, macroValue);
        }
    }
    return compileOptions;
}


ProgramStage* ShaderImporter::compileProgram(const std::string& fileName,
                                             const ProgramStageType& type,
                                             ProgramCompiler::CompileOptions&& compileOptions,
                                             const std::vector<std::string>& includePathList)
{
    auto [fileExist, filePath] = mFileManager->findFile(fileName);
    if (!fileExist) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileNotFound, "fail to open file [{}]", fileName);
    }

    if (auto programStage = mHardwareResourceManager->findProgramStage(filePath); programStage) {
        return programStage;
    }

    auto fileStream = mFileManager->openFile(filePath, FileAccessMode::READ);
    if (!fileStream) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileSystemError, "fail to open file [{}]", filePath);
    }

    ProgramCompiler programCompiler;

    std::unique_ptr<ShaderFileIncluder> shaderFileIncluder = std::make_unique<ShaderFileIncluder>(includePathList);
    programCompiler.setIncluder(std::move(shaderFileIncluder));
    auto binarySource = programCompiler.compile(fileName, fileStream->getAsString(), type, std::move(compileOptions));
    return mHardwareResourceManager->createProgramStage(filePath, std::move(binarySource), type);
}


} // namespace Syrinx
