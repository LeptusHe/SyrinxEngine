#include "ResourceLoader/SyrinxXmlParser.h"
#include "ResourceLoader/SyrinxMaterialParser.h"
#include "ResourceLoader/SyrinxProgramParser.h"
#include <HardwareResource/SyrinxProgramStage.h>
#include <Container/String.h>
#include <Common/SyrinxAssert.h>
#include <Exception/SyrinxException.h>
#include <FileSystem/SyrinxFileSystem.h>

namespace Syrinx {

MaterialParser::MaterialParser(FileManager *fileManager, HardwareResourceManager *hardwareResourceManager, ShaderManager *shaderManager)
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


std::unique_ptr<Material> MaterialParser::parseMaterial(const std::string& fileName)
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

    std::unique_ptr<Material> material;
    std::string shaderFileName;
    try {
        auto materialNode = getChild(document, "material");
        material = std::make_unique<Material>(fileName);
        mMaterial = material.get();
        SYRINX_ASSERT(mMaterial);

        auto shaderFileNode = getChild(materialNode, "shader-file");
        shaderFileName = getText(shaderFileNode);
        Shader *shader = mShaderManager->createOrRetrieve(shaderFileName);
        material->setShader(shader);

        if (auto materialParameterSetNode = materialNode.child("input-parameter-set"); !materialParameterSetNode.empty()) {
            parseMaterialParameterSet(materialParameterSetNode);
        }
    } catch (std::exception& e) {
        clear();
        mShaderManager->remove(shaderFileName);
        SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "fail to parse material [{}] because [{}]", fileName, e.what());
    }
    clear();

    SYRINX_ENSURE(!mMaterial);
    return material;
}


void MaterialParser::parseMaterialParameterSet(const pugi::xml_node& parameterSetNode)
{
    SYRINX_EXPECT(mMaterial && mMaterial->getShader());

    auto shader = mMaterial->getShader();
    for (const auto& parameterNode : parameterSetNode) {
        if (std::string(parameterNode.name()) != "parameter") {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                    "invalid element [{}] in element <input-parameter-set> of shader [{}]", parameterNode.name(), shader->getName());
        }
        std::string name = getAttribute(parameterNode, "name").as_string();
        std::string type = getAttribute(parameterNode, "type").as_string();

        auto *materialParameter = new ShaderParameter();
        materialParameter->setName(name);
        materialParameter->setType(type);
        ShaderParameter *shaderParameter = shader->getShaderParameter(name);
        if (!shaderParameter) {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams, "can not find material parameter [{}] in shader [{}]", name, shader->getName());
        }

        if (shaderParameter->getType() != materialParameter->getType()) {
            SYRINX_THROW_EXCEPTION_FMT(ExceptionCode::InvalidParams,
                    "the type of material parameter [{}] is not the same as the shader parameter [{}] in shader [{}]", name, name, shader->getName());
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


void MaterialParser::clear()
{
    mMaterial = nullptr;
}

} // namespace Syrinx