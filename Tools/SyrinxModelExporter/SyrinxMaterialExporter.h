#pragma once
#include <assimp/material.h>
#include <pugixml.hpp>
#include "SyrinxExporterOptions.h"

namespace Syrinx::Tool {

class MaterialExporter {
public:
    MaterialExporter() = default;
    ~MaterialExporter() = default;
    void exportMaterial(const aiMaterial& material, const std::string& outputFilePath, const ExporterOptions& options) const;

private:
    pugi::xml_node createMaterialElement(const aiMaterial& material, pugi::xml_document *document) const;
    pugi::xml_node createInputParameterSetElement(pugi::xml_node *materialElement) const;
    pugi::xml_node createShaderFileElement(pugi::xml_node *materialElement, const std::string& shaderFileName) const;
    pugi::xml_node createColorParameterElement(pugi::xml_node *inputParameterSetElement, const std::string& name, const std::string& type, const std::string& value) const;
    pugi::xml_node createTextureParameterElement(pugi::xml_node *inputParameterSetElement, const std::string& name, const std::string& textureFileName) const;
    void exportMaterialColors(const aiMaterial& material, pugi::xml_node *inputParameterSetElement) const;
    void exportMaterialTextures(const aiMaterial& material, pugi::xml_node *inputParameterSetElement) const;
};

} // namespace Syrinx::Tool
