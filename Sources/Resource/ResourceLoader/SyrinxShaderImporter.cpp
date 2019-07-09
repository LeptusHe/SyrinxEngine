#include "ResourceLoader/SyrinxShaderImporter.h"
#include <Container/SyrinxString.h>
#include <Exception/SyrinxException.h>
#include <Script/SyrinxLuaScript.h>
#include <Program/SyrinxProgramCompiler.h>

namespace Syrinx {

ShaderImporter::ShaderImporter(FileManager *fileManager,
                               HardwareResourceManager *hardwareResourceManager,
                               ProgramCompiler *compiler)
    : mFileManager(fileManager)
    , mHardwareResourceManager(hardwareResourceManager)
    , mCompiler(compiler)
{
    SYRINX_ENSURE(mFileManager);
    SYRINX_ENSURE(mHardwareResourceManager);
    SYRINX_ENSURE(mCompiler);
}


std::unique_ptr<Shader> ShaderImporter::import(const std::string& fileName)
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
    auto vertexProgramStage = compileProgram(vertexProgramFileName, ProgramStageType::VertexStage);
    auto fragmentProgramStage = compileProgram(fragmentProgramFileName, ProgramStageType::FragmentStage);

    programPipeline->bindProgramStage(vertexProgramStage);
    programPipeline->bindProgramStage(fragmentProgramStage);

    auto shader = std::make_unique<Shader>(shaderName);
    shader->addProgramPipeline(programPipeline);

    return shader;
}


ProgramStage* ShaderImporter::compileProgram(const std::string& fileName, const ProgramStageType& type)
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
    auto binarySource = mCompiler->compile(fileName, fileStream->getAsString(), type);
    return mHardwareResourceManager->createProgramStage(filePath, std::move(binarySource), type);
}


} // namespace Syrinx
