#include <syrinx-vertex-variables.glsl>
#include <syrinx-transform.glsl>

layout (location = 0) out GeometryData {
    vec3 posW;
    vec3 normalW;
    vec3 tangentW;
    vec3 bitangentW;
    vec2 texCoord;
} outGeometryData;


void main()
{
    outGeometryData.posW = SyrinxObjectToWorldPos(inPos).xyz;
    outGeometryData.normalW = SyrinxObjectToWorldNormal(inNormal);
    outGeometryData.texCoord = inTexCoord;

    gl_Position = SyrinxObjectToClipPos(inPos);
}
