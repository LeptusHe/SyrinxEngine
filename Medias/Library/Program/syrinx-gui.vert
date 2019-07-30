layout (std140, binding = 0) uniform SyrinxGuiMatrixBuffer {
    mat4 SYRINX_MATRIX_PROJ;
};

out gl_PerVertex {
    vec4 gl_Position;
};


layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec4 inColor;

layout(location = 0) out vec2 outUV;
layout(location = 1) out vec4 outColor;


void main()
{
    outUV = inUV;
    outColor = inColor;
    gl_Position = SYRINX_MATRIX_PROJ * vec4(inPos, 0.0, 1.0);
}