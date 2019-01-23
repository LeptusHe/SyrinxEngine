#version 450 core

in vec2 _TexCoord;
in mat3 _TBN;

uniform sampler2D uAlbedoMap;
uniform sampler2D uNormalMap;

out vec4 _outFragColor;


void main()
{
    vec4 albedo = texture(uAlbedoMap, _TexCoord);
    vec3 normal = texture(uNormalMap, _TexCoord).xyz;
    normal = normalize(normal * vec3(2.0) - vec3(1.0));
    normal = normalize(_TBN * normal);

    vec3 lightDir = vec3(0, 0, 1);
    vec3 diffuse = dot(normal, lightDir) * vec3(albedo) * vec3(2.0);

    _outFragColor = vec4(diffuse, 1.0);
}
