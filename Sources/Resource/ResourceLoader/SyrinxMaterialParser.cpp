#include "ResourceLoader/SyrinxXmlParser.h"
#include "ResourceLoader/SyrinxMaterialParser.h"
#include "ResourceLoader/SyrinxProgramParser.h"
#include <HardwareResource/SyrinxProgramStage.h>
#include <Container/String.h>
#include <Common/SyrinxAssert.h>
#include <Exception/SyrinxException.h>
#include <FileSystem/SyrinxFileSystem.h>

namespace Syrinx {

MaterialParser::MaterialParser(FileManager *fileManager, HardwareResourceManager *hardwareResourceManager, MaterialManager *materialManager)
    : mFileManager(fileManager)
    , mHardwareResourceManager(hardwareResourceManager)
    , mMaterialManager(materialManager)
    , mMaterial(nullptr)
    , mShader(nullptr)
    , mShaderPass(nullptr)
{
    SYRINX_ENSURE(mFileManager);
    SYRINX_ENSURE(mHardwareResourceManager);
    SYRINX_ENSURE(mMaterialManager);
    SYRINX_ENSURE(!mMaterial);
    SYRINX_ENSURE(!mShader);
    SYRINX_ENSURE(!mShaderPass);
}


Material* MaterialParser::parseMaterial(const std::string& fileName)
{
    auto [fileExist, filePath] = mFileManager->findFile(fileName);
    if (!fileExist) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileNotFound, "can not find file [{}]", fileName);
    }
    auto fileStream = mFileManager->openFile(filePath, FileAccessMode::READ);
    if (!fileStream) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileSystemError, "can not open file [{}]", filePath);
    }

    const std::string fileContent = fileStream->getAsString();
    pugi::xml_document document;
    if (auto result = document.load_string(fileContent.c_str()); !result) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "fail to parse material [{}] because [{}]", fileName, result.description());
    }

    Material *material = nullptr;
    try {
        auto materialNode = getChild(document, "material");
        std::string materialName = getAttribute(materialNode, "name").as_string();
        material = new Material(materialName);
        mMaterial = material;
        SYRINX_ENSURE(mMaterial);

        auto shaderFileNode = getChild(materialNode, "shader-file");
        Shader *shader = parseShader(getText(shaderFileNode));
        material->setShader(shader);

        if (auto materialParameterSetNode = materialNode.child("input-parameter-set"); !materialParameterSetNode.empty()) {
            parseMaterialParameterSet(materialParameterSetNode);
        }
    } catch (std::exception& e) {
        delete material;
        clear();
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "fail to parse material [{}] because [{}]", fileName, e.what());
    }
    clear();

    SYRINX_ENSURE(!mMaterial);
    SYRINX_ENSURE(!mShader);
    SYRINX_ENSURE(!mShaderPass);
    return material;
}


Shader* MaterialParser::parseShader(const std::string& fileName)
{
    if (auto shader = mMaterialManager->findShader(fileName); shader) {
        mShader = shader;
        return shader;
    }

    auto fileStream = mFileManager->openFile(fileName, FileAccessMode::READ);
    if (!fileStream) {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::FileNotFound, "can not open file [{}]", fileName);
    }

    pugi::xml_document shaderDocument;
    std::string fileContent = fileStream->getAsString();
    shaderDocument.load_string(fileContent.c_str());

    auto shaderNode = getChild(shaderDocument, "shader");
    std::string shaderName = getAttribute(shaderNode, "name").as_string();
    mShader = mMaterialManager->createShader(fileName);

    if (auto shaderParameterSetNode = shaderNode.child("input-parameter-set"); !shaderParameterSetNode.empty()) {
        mShader->addShaderParameterSet(parseShaderParameterSet(shaderParameterSetNode));
    }
    mShader->addShaderPassSet(parseShaderPassSet(shaderNode));
    return mShader;
}


void MaterialParser::parseMaterialParameterSet(const pugi::xml_node& parameterSetNode)
{
    SYRINX_EXPECT(mMaterial && mShader);
    SYRINX_EXPECT(mMaterial->getShader() == mShader);
    for (const auto& parameterNode : parameterSetNode) {
        if (std::string(parameterNode.name()) != "parameter") {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                    "invalid element [{}] in element <input-parameter-set> of shader [{}]", parameterNode.name(), mShader->getName());
        }
        std::string name = getAttribute(parameterNode, "name").as_string();
        std::string type = getAttribute(parameterNode, "type").as_string();

        auto *materialParameter = new ShaderParameter();
        materialParameter->setName(name);
        materialParameter->setType(type);
        ShaderParameter *shaderParameter = mShader->getShaderParameter(name);
        if (!shaderParameter) {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "can not find material parameter [{}] in shader [{}]", name, mShader->getName());
        }

        if (shaderParameter->getType() != materialParameter->getType()) {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                    "the type of material parameter [{}] is not the same as the shader parameter [{}] in shader [{}]", name, name, mShader->getName());
        }
        materialParameter->setValue(shaderParameter->getValue());
        parseMaterialParameterValue(parameterNode, *materialParameter);
        mMaterial->addMaterialParameter(materialParameter);
    }
}


void MaterialParser::parseMaterialParameterValue(const pugi::xml_node& parameterNode, ShaderParameter& shaderParameter)
{
    auto parameterType = shaderParameter.getType()._value;
    if (parameterType == ShaderParameterType::INT) {
        shaderParameter.setValue(getAttribute(parameterNode, "value").as_int());
    } else if (parameterType == ShaderParameterType::FLOAT) {
        shaderParameter.setValue(getAttribute(parameterNode, "value").as_float());
    } else if (parameterType == ShaderParameterType::COLOR) {
        const std::string valueString = getAttribute(parameterNode, "value").as_string();
        std::vector<float> floatArray = ParseFloatArray(valueString);
        if (floatArray.size() != 4) {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "invalid value [{}] for color type in material parameter [{}]", valueString, shaderParameter.getName());
        }
        shaderParameter.setValue(Color(floatArray.data()));
    } else if (parameterType == ShaderParameterType::TEXTURE_2D) {
        std::string imageFile = getText(getChild(parameterNode, "image-file"));
        std::string imageFormat = ToUpper(getText(getChild(parameterNode, "image-format")));
        HardwareTexture *hardwareTexture = mHardwareResourceManager->createTexture(imageFile, ImageFormat::_from_string(imageFormat.c_str()));
        auto& textureValue = std::get<TextureValue>(shaderParameter.getValue());
        textureValue.texture = hardwareTexture;
    } else {
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "unsupported shader parameter type [{}]", shaderParameter.getType()._to_string());
    }
}


std::vector<std::unique_ptr<ShaderParameter>> MaterialParser::parseShaderParameterSet(const pugi::xml_node& parameterSetNode)
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


std::unique_ptr<ShaderParameter> MaterialParser::parseShaderParameter(const pugi::xml_node& parameterNode)
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


ShaderParameter::Value MaterialParser::parseShaderParameterValue(const ShaderParameter& shaderParameter, const std::string& valueString)
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


std::vector<std::unique_ptr<ShaderPass>> MaterialParser::parseShaderPassSet(const pugi::xml_node& shaderNode)
{
    std::vector<std::unique_ptr<ShaderPass>> shaderPassSet;
    for (const auto& shaderPassNode : shaderNode.children("pass")) {
        auto shaderPass = parseShaderPass(shaderPassNode);
        shaderPassSet.push_back(std::move(shaderPass));
    }
    return shaderPassSet;
}


std::unique_ptr<ShaderPass> MaterialParser::parseShaderPass(const pugi::xml_node& shaderPassNode)
{
    std::string name = getAttribute(shaderPassNode, "name").as_string();
    auto shaderPass = std::make_unique<ShaderPass>(name);
    mShaderPass = shaderPass.get();
    parseVertexProgram(getChild(shaderPassNode, "vertex-program"), shaderPass.get());
    parseFragmentProgram(getChild(shaderPassNode, "fragment-program"), shaderPass.get());

    auto programPipeline = mHardwareResourceManager->createProgramPipeline("program pipeline for shader pass [" + name + "]");
    programPipeline->bindProgramStage(mShaderPass->getProgramStage(ProgramStageType::VertexStage));
    programPipeline->bindProgramStage(mShaderPass->getProgramStage(ProgramStageType::FragmentStage));
    mShaderPass->setProgramPipeline(programPipeline);

    SYRINX_ENSURE(mShaderPass->getProgramPipeline());
    return shaderPass;
}


void MaterialParser::parseVertexProgram(const pugi::xml_node& vertexProgramNode, ShaderPass *shaderPass)
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


void MaterialParser::parseFragmentProgram(const pugi::xml_node& fragmentProgramNode, ShaderPass *shaderPass)
{
    SYRINX_EXPECT(shaderPass);
    std::string codeFile = getText(getChild(fragmentProgramNode, "code-file"));
    auto fragmentProgram = mHardwareResourceManager->createProgramStage(codeFile, ProgramStageType::FragmentStage);
    shaderPass->addProgramStage(ProgramStageType::FragmentStage, fragmentProgram);

    auto programParameterSet = parseProgramParameterSet(fragmentProgramNode.child("input-parameter-set"));
    shaderPass->addParameterRefSetForFragmentProgram(programParameterSet);
}


std::vector<VertexAttribute> MaterialParser::parseVertexAttributeSet(const pugi::xml_node& vertexAttributeSetNode)
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


std::vector<ShaderParameter*> MaterialParser::parseProgramParameterSet(const pugi::xml_node& programParameterSetNode)
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


void MaterialParser::clear()
{
    mMaterial = nullptr;
    mShader = nullptr;
    mShaderPass = nullptr;
}

} // namespace Syrinx