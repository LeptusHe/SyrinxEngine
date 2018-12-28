#pragma once
#include <string>

namespace Syrinx::Tool {

struct ExporterOptions {
    bool exportMaterialColor = false;
    std::string shaderFileName = "default-shader-file.shader";
};

} // namespace Syrinx::Tool