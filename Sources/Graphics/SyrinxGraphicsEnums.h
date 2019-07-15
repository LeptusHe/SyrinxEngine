#pragma once
#include <better-enums/enum.h>

namespace Syrinx {

BETTER_ENUM(PrimitiveTopology, uint8_t, Point, Line, LineStrip, Triangle, TriangleStrip, TriangleFan);

BETTER_ENUM(PolygonMode, uint8_t, Point, Line, Fill);

BETTER_ENUM(CullMode, uint8_t, None, Front, Back, FrontAndBack);

BETTER_ENUM(BlendFactor, uint8_t,
    Zero,
    One,
    SrcColor,
    DstColor,
    SrcAlpha,
    DstAlpha,
    OneMinusSrcColor,
    OneMinusDstColor,
    OneMinusSrcAlpha,
    OneMinusDstAlpha,
    ConstantColor,
    OneMinusConstantColor,
    ConstantAlpha,
    OneMinusConstantAlpha);

BETTER_ENUM(BlendOp, uint8_t, Add, Subtract, ReverseSubtract, Min, Max);


BETTER_ENUM(TextureType, uint8_t,
    TEXTURE_2D,
    TEXTURE_3D,
    TEXTURE_CUBEMAP,
    TEXTURE_2D_ARRAY,
    UNDEFINED
);


BETTER_ENUM(ProgramStageType, std::uint8_t,
    UndefinedStage,
    VertexStage,
    TessellationControlStage,
    TessellationEvaluationStage,
    GeometryStage,
    FragmentStage,
    ComputeStage
);



BETTER_ENUM(VertexAttributeSemantic, uint8_t,
    Undefined,
    Position,
    Normal,
    TexCoord,
    Tangent,
    Bitangent,
    Color
);


BETTER_ENUM(VertexAttributeDataType, uint8_t,
    Undefined,
    UBYTE1,
    UBYTE2,
    UBYTE3,
    UBYTE4,

    SHORT1,
    SHORT2,
    SHORT3,
    SHORT4,

    FLOAT1,
    FLOAT2,
    FLOAT3,
    FLOAT4,

    DOUBLE1,
    DOUBLE2,
    DOUBLE3,
    DOUBLE4
);

BETTER_ENUM(TextureMinFilterMethod, uint8_t,
    NEAREST,
    LINEAR,
    NEAREST_MIPMAP_NEAREST,
    LINEAR_MIPMAP_NEAREST,
    NEAREST_MIPMAP_LINEAR,
    LINEAR_MIPMAP_LINEAR
);


BETTER_ENUM(TextureMagFilterMethod, uint8_t,
    NEAREST,
    LINEAR
);


BETTER_ENUM(TextureWrapMethod, uint8_t,
    CLAMP_TO_BORDER,
    CLAMP_TO_EDGE,
    MIRROR_CLAMP_TO_EDGE,
    REPEAT,
    MIRRORED_REPEAT
);

} // namespace Syrinx
