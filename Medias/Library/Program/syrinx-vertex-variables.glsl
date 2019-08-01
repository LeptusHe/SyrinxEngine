#ifndef SYRINX_VERTEX_VARIABLES_PROGRAM
#define SYRINX_VERTEX_VARIABLES_PROGRAM

layout (std140, binding = 0) uniform SyrinxMatrixBuffer {
    mat4 SYRINX_MATRIX_WORLD;
    mat4 SYRINX_MATRIX_VIEW;
    mat4 SYRINX_MATRIX_PROJ;
};


layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;


out gl_PerVertex {
    vec4 gl_Position;
};

#endif // SYRINX_VERTEX_VARIABLES_PROGRAM