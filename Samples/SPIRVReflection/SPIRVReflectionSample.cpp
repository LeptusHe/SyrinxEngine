#include <Logging/SyrinxLogManager.h>
#include <FileSystem/SyrinxFileManager.h>
#include <spirv_glsl.hpp>
#include <shaderc/shaderc.hpp>
#include <Logging/SyrinxLogManager.h>
#include <Program/SyrinxProgramReflector.h>
#include <iostream>
#include "SyrinxFileIncluder.h"


void printStructInfo(Syrinx::StructBlockInfo *structBlockInfo, const std::string& indent)
{
    SYRINX_EXPECT(structBlockInfo);
    std::cout << indent
              << "struct block info : [name =" << structBlockInfo->name
              << ", offset =" << structBlockInfo->offset
              << ", size = " << structBlockInfo->size << "]" << std::endl;
    std::cout << indent << "struct member list info: {" << std::endl;
    for (const auto memberInfo : structBlockInfo->memberInfoList) {
        if (memberInfo->type.basetype == Syrinx::ReflectionType::BaseType::Struct) {
            printStructInfo(reinterpret_cast<Syrinx::StructBlockInfo*>(memberInfo), indent + "    ");
        } else {
            std::cout << indent + "    "
                      <<  "name: " << memberInfo->name
                      << ", offset: " << memberInfo->offset
                      << ", size: " << memberInfo->size << std::endl;
        }
    }
}



int main(int argc, char *argv[])
{
    auto logManager = std::make_unique<Syrinx::LogManager>();
    Syrinx::FileManager fileManager;

    fileManager.addSearchPath(".");
    auto [fileExist, fileFullPath] = fileManager.findFile("VertexShader.vert");
    SYRINX_ASSERT(fileExist);
    auto sourceFileStream = fileManager.openFile(fileFullPath, Syrinx::FileAccessMode::READ);

    shaderc::Compiler compiler;
    shaderc::CompileOptions compileOptions;
    compileOptions.SetForcedVersionProfile(450, shaderc_profile_core);
    compileOptions.SetTargetEnvironment(shaderc_target_env_opengl, 0);
    compileOptions.SetSourceLanguage(shaderc_source_language_glsl);
    compileOptions.SetGenerateDebugInfo();
    compileOptions.SetWarningsAsErrors();
    auto fileIncluder = std::make_unique<Syrinx::FileIncluder>();
    fileIncluder->addSearchPath(".");
    compileOptions.SetIncluder(std::move(fileIncluder));

    const std::string sources = sourceFileStream->getAsString();
    auto module = compiler.CompileGlslToSpv(sources, shaderc_glsl_vertex_shader, fileFullPath.c_str(), compileOptions);
    if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
        SYRINX_DEBUG_FMT("compile error: [{}]", module.GetErrorMessage());
    }

    std::vector<uint32_t> byteArray{module.cbegin(), module.cend()};
    Syrinx::ProgramReflector programReflector(std::move(byteArray));
    auto inputInterfaceList = programReflector.getInputInterfaceList();
    for (const auto& input : inputInterfaceList) {
        SYRINX_INFO_FMT("stage input: [name={}, location={}]", input->name, input->location);
    }

    auto sampledTextureList = programReflector.getSampledTextureList();
    for (const auto& sampledTexture : sampledTextureList) {
        SYRINX_INFO_FMT("sampled texture: [name={}, type={}, binding={}]", sampledTexture->name, sampledTexture->type._to_string(), sampledTexture->binding);
    }

    auto uniformBufferList = programReflector.getUniformBufferList();
    for (const auto uniformBuffer : uniformBufferList) {
        SYRINX_INFO_FMT("uniform buffer: [name={}, binding={}, size={}]",
            uniformBuffer->name,
            uniformBuffer->binding,
            uniformBuffer->size);
        printStructInfo(uniformBuffer, "");
    }

    auto matrixStateUniformBuffer = *uniformBufferList[0];
    assert(matrixStateUniformBuffer.isMemberExist("light_pos"));
    auto lightPos = matrixStateUniformBuffer["light_pos"];
    SYRINX_INFO_FMT("light pos: [size={}, offset={}]", lightPos.size, lightPos.offset);

    return 0;
}