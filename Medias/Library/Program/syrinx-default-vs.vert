#include <syrinx-vertex-variables.glsl>
#include <syrinx-transform.glsl>

layout (location = 0) out vec3 outPosWorld;
layout (location = 1) out vec3 outNormalWorld;


void main()
{
    outPosWorld = SyrinxObjectToWorldPos(inPos).xyz;
    outNormalWorld = SyrinxObjectToWorldNormal(inNormal);

    gl_Position = SyrinxObjectToClipPos(inPos);
}
