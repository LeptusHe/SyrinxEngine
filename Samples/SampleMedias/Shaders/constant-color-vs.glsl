#include <syrinx-vertex-variables.glsl>

void main()
{
    gl_Position = SyrinxObjectToClipPos(inPos);
}
