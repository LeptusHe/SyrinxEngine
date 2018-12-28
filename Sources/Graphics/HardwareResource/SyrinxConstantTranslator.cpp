#include "HardwareResource/SyrinxConstantTranslator.h"
#include <Common/SyrinxAssert.h>

namespace Syrinx {

GLenum ConstantTranslator::getProgramStageType(ProgramStageType type) {
    switch (type) {
        case ProgramStageType::VertexStage: return GL_VERTEX_SHADER;
        case ProgramStageType::GeometryStage: return GL_GEOMETRY_SHADER;
        case ProgramStageType::FragmentStage: return GL_FRAGMENT_SHADER;
        case ProgramStageType::ComputeStage: return GL_COMPUTE_SHADER;
        default: SYRINX_ASSERT(false && "undefined program stage type");
    }
    return GL_INVALID_ENUM;
}


GLenum ConstantTranslator::getProgramStageTypeBitfield(ProgramStageType type) {
    switch (type) {
        case ProgramStageType::VertexStage: return GL_VERTEX_SHADER_BIT;
        case ProgramStageType::GeometryStage: return GL_GEOMETRY_SHADER_BIT;
        case ProgramStageType::FragmentStage: return GL_FRAGMENT_SHADER_BIT;
        case ProgramStageType::ComputeStage: return GL_COMPUTE_SHADER_BIT;
        default: SYRINX_ASSERT(false && "undefined program stage type");
    }
    return GL_INVALID_ENUM;
}


std::pair<int, GLenum> ConstantTranslator::getOpenGLValueType(VertexAttributeDataType dataType) {
    switch (dataType) {
        case VertexAttributeDataType::SHORT1:
            return {1, GL_SHORT};
        case VertexAttributeDataType::SHORT2:
            return {2, GL_SHORT};
        case VertexAttributeDataType::SHORT3:
            return {3, GL_SHORT};
        case VertexAttributeDataType::SHORT4:
            return {4, GL_SHORT};
        case VertexAttributeDataType::FLOAT1:
            return {1, GL_FLOAT};
        case VertexAttributeDataType::FLOAT2:
            return {2, GL_FLOAT};
        case VertexAttributeDataType::FLOAT3:
            return {3, GL_FLOAT};
        case VertexAttributeDataType::FLOAT4:
            return {4, GL_FLOAT};
        default:
            SYRINX_ASSERT(false && "undefined data type for vertex attribute");
    }
    return {0, GL_INVALID_ENUM};
}


GLenum ConstantTranslator::getPixelFormat(PixelFormat pixelFormat) {
    switch (pixelFormat) {
        case PixelFormat::RED8: return GL_R8;
        case PixelFormat::RG8: return GL_RG8;
        case PixelFormat::RGB8: return GL_RGB8;
        case PixelFormat::RGBA8: return GL_RGBA8;

        case PixelFormat::REDF: return GL_R32F;
        case PixelFormat::RGF: return GL_RG32F;
        case PixelFormat::RG16F: return GL_RG16F;
        case PixelFormat::RGBF: return GL_RGB32F;
        case PixelFormat::RGB16F: return GL_RGB16F;
        case PixelFormat::RGBAF: return GL_RGBA32F;

        case PixelFormat::DEPTH16: return GL_DEPTH_COMPONENT16;
        case PixelFormat::DEPTH24: return GL_DEPTH_COMPONENT24;
        case PixelFormat::DEPTH32F: return GL_DEPTH_COMPONENT32F;
        default: SYRINX_ASSERT(false && "undefined pixel format");
    }
    return GL_INVALID_ENUM;
}


size_t ConstantTranslator::getSizeOfPixelFormat(PixelFormat pixelFormat)
{
    switch (pixelFormat) {
        case PixelFormat::RED8:  return 1 * sizeof(uint8_t);
        case PixelFormat::RG8:   return 2 * sizeof(uint8_t);
        case PixelFormat::RGB8:  return 3 * sizeof(uint8_t);
        case PixelFormat::RGBA8: return 4 * sizeof(uint8_t);

        case PixelFormat::REDF:   return 1 * sizeof(float);
        case PixelFormat::RGF:    return 2 * sizeof(float);
        case PixelFormat::RG16F:  return 2 * sizeof(float) / 2;
        case PixelFormat::RGBF:   return 3 * sizeof(float);
        case PixelFormat::RGB16F: return 3 * sizeof(float) / 2;
        case PixelFormat::RGBAF:  return 4 * sizeof(float);
        default: SYRINX_ASSERT(false && "undefined pixel format");
    }
    return 0;
}


std::pair<GLenum, GLenum> ConstantTranslator::getPixelComponentAndComponentType(PixelFormat pixelFormat)
{
    switch (pixelFormat) {
        case PixelFormat::RED8: return {GL_RED, GL_UNSIGNED_BYTE};
        case PixelFormat::RG8: return {GL_RG, GL_UNSIGNED_BYTE};
        case PixelFormat::RGB8: return {GL_RGB, GL_UNSIGNED_BYTE};
        case PixelFormat::RGBA8: return {GL_RGBA, GL_UNSIGNED_BYTE};

        case PixelFormat::REDF: return {GL_RED, GL_FLOAT};
        case PixelFormat::RGF: return {GL_RG, GL_FLOAT};
        case PixelFormat::RG16F: return {GL_RG, GL_FLOAT};
        case PixelFormat::RGBF: return {GL_RGB, GL_FLOAT};
        case PixelFormat::RGB16F: return {GL_RGB, GL_FLOAT};
        case PixelFormat::RGBAF: return {GL_RGBA, GL_FLOAT};
        default: SYRINX_ASSERT(false && "undefined pixel format");
    }
    return {GL_INVALID_ENUM, GL_INVALID_ENUM};
}


GLenum ConstantTranslator::getTextureType(TextureType textureType)
{
    switch (textureType) {
        case TextureType::TEXTURE_2D: return GL_TEXTURE_2D;
        case TextureType::TEXTURE_CUBEMAP: return GL_TEXTURE_CUBE_MAP;
        default: SYRINX_ASSERT(false && "undefined texture type");
    }
    return GL_INVALID_ENUM;
}


GLint ConstantTranslator::getTextureMinFilterMethod(TextureMinFilterMethod textureMinFilterMethod)
{
    switch (textureMinFilterMethod) {
        case TextureMinFilterMethod::NEAREST: return GL_NEAREST;
        case TextureMinFilterMethod::LINEAR: return GL_LINEAR;
        case TextureMinFilterMethod::NEAREST_MIPMAP_NEAREST: return GL_NEAREST_MIPMAP_NEAREST;
        case TextureMinFilterMethod::LINEAR_MIPMAP_NEAREST: return GL_LINEAR_MIPMAP_NEAREST;
        case TextureMinFilterMethod::NEAREST_MIPMAP_LINEAR: return GL_NEAREST_MIPMAP_LINEAR;
        case TextureMinFilterMethod::LINEAR_MIPMAP_LINEAR: return GL_LINEAR_MIPMAP_LINEAR;
        default: SYRINX_ASSERT(false && "undefined texture min filter method");
    }
    return GL_INVALID_ENUM;
}


GLint ConstantTranslator::getTextureMagFilterMethod(TextureMagFilterMethod textureMagFilterMethod)
{
    switch (textureMagFilterMethod) {
        case TextureMagFilterMethod::NEAREST: return GL_NEAREST;
        case TextureMagFilterMethod::LINEAR: return GL_LINEAR;
        default: SYRINX_ASSERT(false && "undefined texture mag filter method");
    }
    return GL_INVALID_ENUM;
}


GLint ConstantTranslator::getTextureWrapMethod(TextureWrapMethod textureWrapMethod)
{
    switch (textureWrapMethod) {
        case TextureWrapMethod::CLAMP_TO_BORDER: return GL_CLAMP_TO_BORDER;
        case TextureWrapMethod::CLAMP_TO_EDGE: return GL_CLAMP_TO_EDGE;
        case TextureWrapMethod::MIRROR_CLAMP_TO_EDGE: return GL_MIRROR_CLAMP_TO_EDGE;
        case TextureWrapMethod::REPEAT: return GL_REPEAT;
        case TextureWrapMethod::MIRRORED_REPEAT: return GL_MIRRORED_REPEAT;
        default: SYRINX_ASSERT(false && "undefined texture wrap method");
    }
    return GL_INVALID_ENUM;
}

} // namespace Syrinx
