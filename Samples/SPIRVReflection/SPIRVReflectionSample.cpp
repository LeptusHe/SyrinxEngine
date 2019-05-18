#include <Logging/SyrinxLogManager.h>
#include <ResourceManager/SyrinxFileManager.h>
#include <spirv_glsl.hpp>
#include <Logging/SyrinxLogManager.h>

int main(int argc, char *argv[])
{
    Syrinx::LogManager *logManager = new Syrinx::LogManager();
    Syrinx::FileManager fileManager;

    fileManager.addSearchPath(".");
    auto fileStream = fileManager.openFile("VertexShaderUnActive.spv", Syrinx::FileAccessMode::READ);
    SYRINX_ASSERT(fileStream);
    auto byteArray = fileStream->getAsDataArray<uint32_t>();

    spirv_cross::CompilerGLSL glslShader(std::move(byteArray));
    auto activeInterfaceVariables = glslShader.get_active_interface_variables();
    auto shaderResources = glslShader.get_shader_resources(activeInterfaceVariables);

    // stage inputs
    for (const auto& stageInput : shaderResources.stage_inputs) {
        auto location = glslShader.get_decoration(stageInput.id, spv::DecorationLocation);
        SYRINX_INFO_FMT("stage input: [name={}, location={}]", stageInput.name, location);
    }

    // stage outputs
    for (const auto& stageOutput : shaderResources.stage_outputs) {
        auto location = glslShader.get_decoration(stageOutput.id, spv::DecorationLocation);
        SYRINX_INFO_FMT("stage output: [name={}, location = {}]", stageOutput.name, location);
    }

    // uniform values
    for (const auto& uniformBuffer : shaderResources.uniform_buffers) {
        auto binding = glslShader.get_decoration(uniformBuffer.id, spv::DecorationBinding);
        SYRINX_INFO_FMT("uniform buffer: [name={}, binding={}]", uniformBuffer.name, binding);
    }

    // push constants
    for (const auto& uniformValue : shaderResources.push_constant_buffers) {
        SYRINX_INFO_FMT("uniform value: [name={}]", uniformValue.name);
    }

    return 0;
}