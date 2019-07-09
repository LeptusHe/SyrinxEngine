#include "DefaultVS.glsl"

struct Falloff {
    float distance;
    float attention;
};

struct Intensity {
    vec3 diffuseIntensity;
    vec3 specularIntensity;
    Falloff falloff;
};

struct LightInfo {
    uint type;
    Intensity intensity;
};

layout(std140, binding = 0) uniform matrix_state {
    mat4 vmat;
    mat4 projmat;
    mat4 mvmat;
    mat4 mvpmat;
    vec3 light_pos;
    LightInfo light;
} matrix;


uniform sampler2D diffuseTex;
uniform sampler2D specTex;
uniform samplerCube cubemapTex;

layout(location = 3) out vec3 vpos;
layout(location = 4) out vec3 norm;
layout(location = 5) out vec3 ldir;
layout(location = 6) out vec2 texcoord;
layout(location = 7) out vec4 difColor;



void main()
{
    gl_Position = matrix.mvpmat * attr_vertex;
    vpos = (matrix.mvmat * attr_vertex).xyz;
    norm = mat3(matrix.mvmat) * attr_normal;
    texcoord = attr_texcoord * vec2(2.0, 1.0);
    ldir = matrix.light_pos - vpos;

    vec4 diffuseColor =texture(diffuseTex, texcoord);
    vec4 cubemap = texture(cubemapTex, vec3(texcoord, 1.0));
    difColor = diffuseColor + cubemap;

}