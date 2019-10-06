#ifndef SYRINX_FRAGMENT_VARIABLES_PROGRAM
#define SYRINX_FRAGMENT_VARIABLES_PROGRAM

layout(location = 0) in GeometryData {
    vec3 posW;
    vec3 normalW;
    vec3 tangentW;
    vec3 bitangentW;
    vec2 texCoord;
} inGeometryData;

#endif // SYRINX_FRAGMENT_VARIABLES_PROGRAM