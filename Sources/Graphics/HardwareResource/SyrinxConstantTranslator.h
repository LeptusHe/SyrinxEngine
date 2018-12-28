#pragma once
#include <utility>
#include <GL/glew.h>
#include "HardwareResource/SyrinxProgramStage.h"
#include "HardwareResource/SyrinxHardwareTexture.h"
#include "HardwareResource/SyrinxVertexAttributeDescription.h"

namespace Syrinx {

class ConstantTranslator {
public:
    static GLenum getProgramStageType(ProgramStageType type);
    static GLenum getProgramStageTypeBitfield(ProgramStageType type);
    static std::pair<int, GLenum> getOpenGLValueType(VertexAttributeDataType dataType);
    static GLenum getPixelFormat(PixelFormat pixelFormat);
    static size_t getSizeOfPixelFormat(PixelFormat pixelFormat);
    static std::pair<GLenum, GLenum> getPixelComponentAndComponentType(PixelFormat pixelFormat);
    static GLenum getTextureType(TextureType textureType);
    static GLint getTextureMinFilterMethod(TextureMinFilterMethod textureMinFilterMethod);
    static GLint getTextureMagFilterMethod(TextureMagFilterMethod textureMagFilterMethod);
    static GLint getTextureWrapMethod(TextureWrapMethod textureWrapMethod);
};

} // namespace Syrinx
