#include <syrinx-vertex-variables.glsl>



void main()
{
    //gl_Position = projMat * viewMat * modelMat * vec4(inPos, 1.0);
    gl_Position = SyrinxObjectToClipPos(inPos);
}
