#include "ResourceLoader/SyrinxShaderParser.h"
#include <Container/String.h>
#include <Exception/SyrinxException.h>
#include "ResourceLoader/SyrinxXmlParser.h"

namespace Syrinx {

ShaderParser::ShaderParser(FileManager *fileManager, HardwareResourceManager *hardwareResourceManager)
    : mFileManager(fileManager)
    , mHardwareResourceManager(hardwareResourceManager)
    , mShader(nullptr)
    , mShaderPass(nullptr)
{
    SYRINX_ENSURE(mFileManager);
    SYRINX_ENSURE(mHardwareResourceManager);
    SYRINX_ENSURE(!mShader);
    SYRINX_ENSURE(!mShaderPass);
}


std::unique_ptr<Shader> ShaderParser::parseShader(const std::string& fileName)
{
    auto fileStream = mFileManager->openFile(fileName, FileAccessMode::READ);
    if (!fileStream) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileNotFound, "can not open file [{}]", fileName);
    }

    pugi::xml_document shaderDocument;
    std::string fileContent = fileStream->getAsString();
    shaderDocument.load_string(fileContent.c_str());

    auto shaderNode = getChild(shaderDocument, "shader");
    std::string shaderName = getAttribute(shaderNode, "name").as_string();

    std::unique_ptr<Shader> shader = std::make_unique<Shader>(fileName);
    mShader = shader.get();
    SYRINX_ASSERT(mShader);

    if (auto shaderParameterSetNode = shaderNode.child("input-parameter-set"); !shaderParameterSetNode.empty()) {
        shader->addShaderParameterSet(parseShaderParameterSet(shaderParameterSetNode));
    }
    shader->addShaderPassSet(parseShaderPassSet(shaderNode));
    clear();

    SYRINX_ENSURE(shader);
    SYRINX_ENSURE(!mShader);
    SYRINX_ENSURE(!mShaderPass);
    return shader;
}


std::vector<std::unique_ptr<ShaderParameter>> ShaderParser::parseShaderParameterSet(const pugi::xml_node& parameterSetNode)
{
    std::vector<std::unique_ptr<ShaderParameter>> shaderParameterSet;
    size_t textureParameterCounter = 0;
    for (const auto& parameterNode : parameterSetNode) {
        if (std::string(parameterNode.name()) != "parameter") {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                                       "invalid element [{}] in element <input-parameter-set> of shader [{}]", parameterNode.name(), mShader->getName());
        }
        auto shaderParameter = parseShaderParameter(parameterNode);
        const std::string parameterType = shaderParameter->getType()._to_string();
        if (parameterType.find("TEXTURE") != std::string::npos) {
            auto& value = std::get<TextureValue>(shaderParameter->getValue());
            value.textureUnit = textureParameterCounter;
            textureParameterCounter += 1;
        }
        shaderParameterSet.push_back(std::move(shaderParameter));
    }
    return shaderParameterSet;
}


std::unique_ptr<ShaderParameter> ShaderParser::parseShaderParameter(const pugi::xml_node& parameterNode)
{
    SYRINX_EXPECT(!parameterNode.empty());
    std::string name = getAttribute(parameterNode, "name").as_string();
    std::string type = getAttribute(parameterNode, "type").as_string();
    std::string value = getAttribute(parameterNode, "value").as_string();

    auto shaderParameter = std::make_unique<ShaderParameter>();
    shaderParameter->setName(name);
    shaderParameter->setType(type);
    shaderParameter->setValue(parseShaderParameterValue(*shaderParameter, value));
    return shaderParameter;
}


ShaderParameter::Value ShaderParser::parseShaderParameterValue(const ShaderParameter& shaderParameter, const std::string& valueString)
{
    auto parameterType = shaderParameter.getType()._value;
    if (parameterType == ShaderParameterType::INT) {
        return std::stoi(valueString);
    } else if (parameterType == ShaderParameterType::FLOAT) {
        return std::stof(valueString);
    } else if (parameterType == ShaderParameterType::COLOR) {
        std::vector<float> floatArray = ParseFloatArray(valueString);
        if (floatArray.size() != 4) {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "invalid value [{}] for color type in shader parameter [{}]", valueString, shaderParameter.getName());
        }
        return Color(floatArray.data());
    } else if (parameterType == ShaderParameterType::TEXTURE_2D) {
        return TextureValue();
    } else {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "unsupported shader parameter type [{}]", shaderParameter.getType()._to_string());
        return 0;
    }
}


std::vector<std::unique_ptr<ShaderPass>> ShaderParser::parseShaderPassSet(const pugi::xml_node& shaderNode)
{
    std::vector<std::unique_ptr<ShaderPass>> shaderPassSet;
    for (const auto& shaderPassNode : shaderNode.children("pass")) {
        auto shaderPass = parseShaderPass(shaderPassNode);
        shaderPassSet.push_back(std::move(shaderPass));
    }
    return shaderPassSet;
}


std::unique_ptr<ShaderPass> ShaderParser::parseShaderPass(const pugi::xml_node& shaderPassNode)
{
    const std::string name = getAttribute(shaderPassNode, "name").as_string();
    auto shaderPass = std::make_unique<ShaderPass>(name);
    mShaderPass = shaderPass.get();
    SYRINX_ASSERT(mShaderPass);

    parseVertexProgram(getChild(shaderPassNode, "vertex-program"), shaderPass.get());
    parseFragmentProgram(getChild(shaderPassNode, "fragment-program"), shaderPass.get());

    auto programPipeline = mHardwareResourceManager->createProgramPipeline("program pipeline for shader pass [" + name + "]");
    programPipeline->bindProgramStage(shaderPass->getProgramStage(ProgramStageType::VertexStage));
    programPipeline->bindProgramStage(shaderPass->getProgramStage(ProgramStageType::FragmentStage));
    shaderPass->setProgramPipeline(programPipeline);

    SYRINX_ENSURE(shaderPass->getProgramPipeline());
    return shaderPass;
}


void ShaderParser::parseVertexProgram(const pugi::xml_node& vertexProgramNode, ShaderPass *shaderPass)
{
    SYRINX_EXPECT(shaderPass);
    std::string codeFile = getText(getChild(vertexProgramNode, "code-file"));
    auto vertexProgram = mHardwareResourceManager->createProgramStage(codeFile, ProgramStageType::VertexStage);
    shaderPass->addProgramStage(ProgramStageType::VertexStage, vertexProgram);

    auto vertexAttributeSet = parseVertexAttributeSet(getChild(vertexProgramNode, "input-vertex-attribute-set"));
    shaderPass->addVertexAttributeSet(vertexAttributeSet);

    auto programParameterSet = parseProgramParameterSet(vertexProgramNode.child("input-parameter-set"));
    shaderPass->addParameterRefSetForVertexProgram(programParameterSet);
}


void ShaderParser::parseFragmentProgram(const pugi::xml_node& fragmentProgramNode, ShaderPass *shaderPass)
{
    SYRINX_EXPECT(shaderPass);
    std::string codeFile = getText(getChild(fragmentProgramNode, "code-file"));
    auto fragmentProgram = mHardwareResourceManager->createProgramStage(codeFile, ProgramStageType::FragmentStage);
    shaderPass->addProgramStage(ProgramStageType::FragmentStage, fragmentProgram);

    auto programParameterSet = parseProgramParameterSet(fragmentProgramNode.child("input-parameter-set"));
    shaderPass->addParameterRefSetForFragmentProgram(programParameterSet);
}


std::vector<VertexAttribute> ShaderParser::parseVertexAttributeSet(const pugi::xml_node& vertexAttributeSetNode)
{
    std::vector<VertexAttribute> vertexAttributeSet;
    for (const auto& vertexAttributeNode : vertexAttributeSetNode) {
        if (std::string(vertexAttributeNode.name()) != "attribute") {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "invalid element name [{}] for parameter", vertexAttributeNode.name());
        }
        std::string name = getAttribute(vertexAttributeNode, "name").as_string();
        std::string semantic = getAttribute(vertexAttributeNode, "semantic").as_string();
        std::string dataType = getAttribute(vertexAttributeNode, "data-type").as_string();

        VertexAttribute vertexAttribute;
        vertexAttribute.setName(name);
        vertexAttribute.setSemantic(semantic);
        vertexAttribute.setDataType(dataType);

        vertexAttributeSet.push_back(vertexAttribute);
    }

    if (vertexAttributeSet.empty()) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "no vertex attribute exist in shade pass [{}]", mShaderPass->getName());
    }
    return vertexAttributeSet;
}


std::vector<ShaderParameter*> ShaderParser::parseProgramParameterSet(const pugi::xml_node& programParameterSetNode)
{
    std::vector<ShaderParameter*> programParameterSet;
    for (const auto& parameterNode : programParameterSetNode) {
        if (std::string(parameterNode.name()) != "parameter") {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "invalid element name [{}] for parameter", parameterNode.name());
        }
        std::string parameterReferenced = getAttribute(parameterNode, "ref").as_string();

        SYRINX_ASSERT(mShader);
        auto shaderParameter = mShader->getShaderParameter(parameterReferenced);
        if (!shaderParameter) {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "fail to reference parameter [{}] in shader pass [{}] because it is not defined", parameterReferenced, mShaderPass->getName());
        }
        programParameterSet.push_back(shaderParameter);
    }
    return programParameterSet;
}


void ShaderParser::clear()
{
    mShader = nullptr;
    mShaderPass = nullptr;
}


} // namespace Syrinx
