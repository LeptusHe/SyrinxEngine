#include <syrinx-constants.glsl>
#include <syrinx-fragment-variables.glsl>

layout(location = 0) out vec4 outColor;


void main()
{
    outColor = SYRINX_PROGRAM_ERROR_COLOR;

#if defined(SYRINX_DISPLAY_WORLD_POSITION)
    outColor = vec4(inGeometryData.posW, 1.0);

#elif defined(SYRINX_DISPLAY_WORLD_NORMAL)
    vec3 normal = inGeometryData.normalW * 0.5 + 0.5;
    outColor = vec4(normal, 1.0);

#else
    #error "display invalid vertex attribute"
#endif
}