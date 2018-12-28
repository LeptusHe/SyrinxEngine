#include "SyrinxMaterialExporter.h"
#include <Common/SyrinxAssert.h>

namespace Syrinx::Tool {

void MaterialExporter::exportMaterial(const aiMaterial& material, const std::string& outputFilePath, const ExporterOptions& options) const
{
    pugi::xml_document document;
    pugi::xml_node materialElement = createMaterialElement(material, &document);
    pugi::xml_node inputParameterSetElement = createInputParameterSetElement(&materialElement);
    pugi::xml_node shaderFileElement = createShaderFileElement(&materialElement, options.shaderFileName);

    if (options.exportMaterialColor) {
        exportMaterialColors(material, &inputParameterSetElement);
    }
    exportMaterialTextures(material, &inputParameterSetElement);

    document.save_file(outputFilePath.c_str(), "    ", pugi::format_indent| pugi::format_no_declaration);
}


pugi::xml_node MaterialExporter::createMaterialElement(const aiMaterial& material, pugi::xml_document *document) const
{
    SYRINX_EXPECT(document);
    pugi::xml_node materialElement = document->append_child("material");
    pugi::xml_attribute nameAttribute = materialElement.append_attribute("name");
    aiString materialName;
    material.Get(AI_MATKEY_NAME, materialName);
    nameAttribute.set_value(materialName.C_Str());
    return materialElement;
}


pugi::xml_node MaterialExporter::createInputParameterSetElement(pugi::xml_node *materialElement) const
{
    SYRINX_EXPECT(materialElement);
    pugi::xml_node inputParameterSetElement = materialElement->append_child("input-parameter-set");
    return inputParameterSetElement;
}


pugi::xml_node MaterialExporter::createShaderFileElement(pugi::xml_node *materialElement, const std::string& shaderFileName) const
{
    SYRINX_EXPECT(materialElement);
    pugi::xml_node shaderFileElement = materialElement->append_child("shader-file");
    shaderFileElement.text() = shaderFileName.c_str();
    return shaderFileElement;
}


pugi::xml_node MaterialExporter::createColorParameterElement(pugi::xml_node *inputParameterSetElement, const std::string& name, const std::string& type, const std::string& value) const
{
    SYRINX_EXPECT(inputParameterSetElement);
    pugi::xml_node parameterElement = inputParameterSetElement->append_child("parameter");
    pugi::xml_attribute nameAttribute = parameterElement.append_attribute("name");
    nameAttribute.set_value(name.c_str());
    pugi::xml_attribute typeAttribute = parameterElement.append_attribute("type");
    typeAttribute.set_value(type.c_str());
    pugi::xml_attribute valueAttribute = parameterElement.append_attribute("value");
    valueAttribute.set_value(value.c_str());
    return parameterElement;
}


pugi::xml_node MaterialExporter::createTextureParameterElement(pugi::xml_node *inputParameterSetElement, const std::string& name, const std::string& textureFileName) const
{
    SYRINX_EXPECT(inputParameterSetElement);
    SYRINX_EXPECT(!name.empty() && !textureFileName.empty());
    pugi::xml_node parameterElement = inputParameterSetElement->append_child("parameter");

    auto nameAttribute = parameterElement.append_attribute("name");
    nameAttribute.set_value(name.c_str());
    auto typeAttribute = parameterElement.append_attribute("type");
    typeAttribute.set_value("texture-2d");

    pugi::xml_node imageFileElement = parameterElement.append_child("image-file");
    imageFileElement.text() = textureFileName.c_str();

    pugi::xml_node imageFormatElement = parameterElement.append_child("image-format");
    imageFormatElement.text() = "rbga8";

    return parameterElement;
}


void MaterialExporter::exportMaterialColors(const aiMaterial& material, pugi::xml_node *inputParameterSetElement) const
{
    SYRINX_EXPECT(inputParameterSetElement);
    auto addColorElement = [&, inputParameterSetElement, this](const char *key, unsigned int type, unsigned int index, const std::string& colorName) {
        aiColor3D color(0.0f, 0.0f, 0.0f);
        if (AI_SUCCESS == material.Get(key, type, index, color)) {
            std::string valueString = std::to_string(color.r) + " " +
                                      std::to_string(color.g) + " " +
                                      std::to_string(color.b) + " " +
                                      std::to_string(1.0);
            pugi::xml_node colorParameterElement = createColorParameterElement(inputParameterSetElement, colorName, "color", valueString);
            inputParameterSetElement->append_move(colorParameterElement);
        }
    };
    addColorElement(AI_MATKEY_COLOR_AMBIENT, "ambientColor");
    addColorElement(AI_MATKEY_COLOR_DIFFUSE, "diffuseColor");
    addColorElement(AI_MATKEY_COLOR_SPECULAR, "specularColor");
}


void MaterialExporter::exportMaterialTextures(const aiMaterial& material, pugi::xml_node *inputParameterSetElement) const
{
    SYRINX_EXPECT(inputParameterSetElement);
    auto addTextureElement = [&, inputParameterSetElement, this](aiTextureType textureType, const std::string textureName) {
        unsigned int textureCount = material.GetTextureCount(textureType);
        for (unsigned int i = 0; i < textureCount; ++ i) {
            aiString texturePath;
            if (material.GetTexture(textureType, i, &texturePath) == AI_SUCCESS) {
                pugi::xml_node textureParameterElement = createTextureParameterElement(inputParameterSetElement, textureName, texturePath.C_Str());
                inputParameterSetElement->append_move(textureParameterElement);
            }
        }
    };

    addTextureElement(aiTextureType_AMBIENT, "ambientTex");
    addTextureElement(aiTextureType_DIFFUSE, "diffuseTex");
    addTextureElement(aiTextureType_SPECULAR, "specularTex");
    addTextureElement(aiTextureType_HEIGHT, "normalTex");
    addTextureElement(aiTextureType_NORMALS, "normal2Tex");
    addTextureElement(aiTextureType_DISPLACEMENT, "displacementTex");
    addTextureElement(aiTextureType_OPACITY, "opacityTex");
}

} // namespace Syrinx
