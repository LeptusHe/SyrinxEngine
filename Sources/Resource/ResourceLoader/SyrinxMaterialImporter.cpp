#include "SyrinxMaterialImporter.h"
#include <Container/SyrinxString.h>
#include <Common/SyrinxAssert.h>
#include <Exception/SyrinxException.h>
#include <FileSystem/SyrinxFileSystem.h>
#include <Script/SyrinxLuaScript.h>

namespace Syrinx {

namespace {

std::pair<std::string, ProgramStageType> kProgramStageMaps[] = {
    {"vertex_program_parameters", ProgramStageType::VertexStage},
    {"fragment_program_parameters", ProgramStageType::FragmentStage}
};


} // anonymous namespace


MaterialImporter::MaterialImporter(FileManager *fileManager,
                                   HardwareResourceManager *hardwareResourceManager,
                                   ShaderManager *shaderManager)
    : mFileManager(fileManager)
    , mHardwareResourceManager(hardwareResourceManager)
    , mShaderManager(shaderManager)
    , mMaterial(nullptr)
{
    SYRINX_ENSURE(mFileManager);
    SYRINX_ENSURE(mHardwareResourceManager);
    SYRINX_ENSURE(mShaderManager);
    SYRINX_ENSURE(!mMaterial);
}


std::unique_ptr<Material> MaterialImporter::import(const std::string& fileName)
{
    auto[fileExist, filePath] = mFileManager->findFile(fileName);
    if (!fileExist) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileNotFound, "can not find file [{}]", fileName);
    }

    sol::state luaState;
    registerVariables(&luaState);
    luaState.script_file(filePath);

    auto material = std::make_unique<Material>(fileName);
    mMaterial = material.get();
    sol::table materialDesc = luaState["material"];
    for (const auto& kvPair : materialDesc) {
        sol::table shaderParameters = kvPair.second;
        std::unique_ptr<ShaderVars> shaderVars = parseShaderVars(shaderParameters);
        material->addShaderVars(std::move(shaderVars));
    }

    mMaterial = nullptr;
    SYRINX_ENSURE(!mMaterial);
    return material;
}


void MaterialImporter::registerVariables(sol::state *state) const
{
    SYRINX_EXPECT(state);
    sol::state& luaState = *state;

    luaState.new_usertype<int>("int");
    luaState.new_usertype<float>("float");
    luaState.new_usertype<glm::vec2>("vec2");
    luaState.new_usertype<glm::vec3>("vec3");
    luaState.new_usertype<glm::vec4>("vec4");
    luaState.new_usertype<glm::ivec2>("ivec2");
    luaState.new_usertype<glm::ivec3>("ivec3");
    luaState.new_usertype<glm::ivec4>("ivec4");
    luaState.new_usertype<glm::uvec2>("uvec2");
    luaState.new_usertype<glm::uvec3>("uvec3");
    luaState.new_usertype<glm::uvec4>("uvec4");
}


std::unique_ptr<ShaderVars> MaterialImporter::parseShaderVars(const sol::table& shaderParameters)
{
    const std::string shaderFileName = shaderParameters["file_name"];
    auto shader = mShaderManager->createOrRetrieve(shaderFileName);
    auto shaderVars = std::make_unique<ShaderVars>(shader);

    for (const auto& kvPair : shaderParameters) {
        const auto& key = kvPair.first;
        const auto& value = kvPair.second;
        std::string keyName = key.as<std::string>();

        for (const auto& stagePair : kProgramStageMaps) {
            if (keyName == stagePair.first) {
                auto programVars = shaderVars->getProgramVars(stagePair.second);
                parseProgramParameters(value.as<sol::table>(), programVars);
            }
        }
    }
    return shaderVars;
}


void MaterialImporter::parseProgramParameters(const sol::table& programParameters, ProgramVars *programVars)
{
    SYRINX_EXPECT(programVars);

    for (const auto& kvPair : programParameters) {
        const std::string parameterName = kvPair.first.as<std::string>();
        sol::table parameterDesc = kvPair.second.as<sol::table>();

        auto uniformBufferInfo = programVars->getUniformBuffer(parameterName);
        auto textureInfo = programVars->getTextureInfo(parameterName);
        if (uniformBufferInfo && textureInfo) {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                       "program [{}] has texture and uniform buffer parameters with the same name [{}] in material [{}]",
                                       programVars->getProgramName(), parameterName, mMaterial->getName());
        }

        if (!uniformBufferInfo && !textureInfo) {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                       "program [{}] doesn't have parameter [{}]",
                                       programVars->getProgramName(), parameterName);
        }

        if (uniformBufferInfo) {
            parseUniformBuffer(parameterName, parameterDesc, programVars);
        } else if (textureInfo) {
            parseTexture2D(parameterName, parameterDesc, programVars);
        } else {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                       "parameter [{}] with invalid parameter type in material [{}]",
                                       parameterName, mMaterial->getName());
        }
    }
}


void MaterialImporter::parseTexture2D(const std::string& texName, const sol::table& value, ProgramVars *programVars)
{
    SYRINX_EXPECT(programVars && mMaterial);
}


void MaterialImporter::parseUniformBuffer(const std::string& uniformBufferName, const sol::table& value, ProgramVars *programVars)
{
    SYRINX_EXPECT(programVars && mMaterial);
    auto uniformBufferInfo = programVars->getUniformBuffer(uniformBufferName);
    if (!uniformBufferInfo) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                   "invalid parameter [{}] in material [{}]",
                                   uniformBufferName, mMaterial->getName());
    }

    auto& uniformBuffer = *uniformBufferInfo;
    parseStructInfo(uniformBufferName, value, uniformBufferInfo);
}


void MaterialImporter::parseStructInfo(const std::string& structName, const sol::table& structDesc, StructBlockInfo *structBlockInfo)
{
    if ((!structBlockInfo) || (structBlockInfo->name != structName)) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                   "invalid struct [{}] in material [{}]",
                                   structName, mMaterial->getName());
    }

    for (const auto& memberNameValue : structDesc) {
        const std::string memberName = memberNameValue.first.as<std::string>();
        const sol::object & memberValue = memberNameValue.second;
        StructMemberInfo *structMemberInfo = structBlockInfo->getMember(memberName);
        if (!structMemberInfo) {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                       "invalid struct member [{}] in material [{}]",
                                       memberName, mMaterial->getName());
        }

        if (structMemberInfo->type.basetype == ReflectionType::BaseType::Struct) {
            parseStructInfo(memberName, memberValue.as<sol::table>(), reinterpret_cast<StructBlockInfo*>(structMemberInfo));
        } else {
            parseVariable(memberName, memberValue, structMemberInfo);
        }
    }
}


void MaterialImporter::parseVariable(const std::string& variableName, const sol::object& value, StructMemberInfo *variableInfo)
{
    if ((!variableInfo) || (variableName != variableInfo->name)) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                  "invalid struct member [{}] in material [{}]",
                                  variableName, mMaterial->getName());
    }

    auto& variable = *variableInfo;
    auto type = variableInfo->type;
    if (type.basetype == ReflectionType::BaseType::Int) {
        variable = value.as<int32_t>();
    } else if (type.basetype == ReflectionType::BaseType::UInt) {
        variable = value.as<uint32_t>();
    } else if (type.basetype == ReflectionType::BaseType::Float && type.vecsize == 1 && type.columns == 1) {
        variable = value.as<float>();
    } else if (type.basetype == ReflectionType::BaseType::Float && type.vecsize == 3 && type.columns == 1) {
        auto vec3Value = value.as<sol::table>();
        float x = vec3Value[1];
        float y = vec3Value[2];
        float z = vec3Value[3];
        variable = glm::vec3(x, y, z);
    } else {
        SHOULD_NOT_GET_HERE();
    }
}

} // namespace Syrinx