#include <syrinx-vertex-variables.glsl>

layout (location = 0) out vec3 outNormal;

void main()
{
    outNormal = inNormal;
    gl_Position = SyrinxObjectToClipPos(inPos);
}
