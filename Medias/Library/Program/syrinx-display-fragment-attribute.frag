#include <syrinx-constants.glsl>

layout (location = 0) in vec3 inPosWorld;
layout (location = 1) in vec3 inNormalWorld;

layout (location = 0) out vec4 outColor;


void main()
{
    outColor = SYRINX_PROGRAM_ERROR_COLOR;

#if defined(SYRINX_DISPLAY_WORLD_POSITION)
    outColor = vec4(inPosWorld, 1.0);

#elif defined(SYRINX_DISPLAY_WORLD_NORMAL)
    vec3 normal = inNormalWorld * 0.5 + 0.5;
    outColor = vec4(normal, 1.0);

#else
    #error "display invalid vertex attribute"
#endif
}