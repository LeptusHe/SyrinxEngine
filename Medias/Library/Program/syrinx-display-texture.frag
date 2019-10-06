#include <syrinx-fragment-variables.glsl>

layout (binding = 0) uniform sampler2D uTex;
layout (location = 0) out vec4 outColor;


void main()
{
    vec4 texColor = texture(uTex, inGeometryData.texCoord);
    outColor = texColor;
}
