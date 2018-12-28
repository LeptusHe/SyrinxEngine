#version 430 core

uniform sampler2D uTextureDiffuse0;
uniform sampler2D uTextureSpecular0;

uniform vec3 uLightDirectionW = vec3(0.0f, 0.0f, -1.0f);
uniform vec3 uLightSpecular = vec3(0.9f, 0.9f, 0.9f);
uniform vec3 uLightDiffuse = vec3(0.5f, 0.5f, 0.5f);
uniform vec3 uEyePosW;

in vec3 _NormalW;
in vec2 _TexCoord;
in vec3 _PositionW;

out vec4 _outFragColor;

void main()
{
    vec3 DiffuseColor = texture(uTextureDiffuse0, _TexCoord).rgb;
    vec3 SpecularColor = texture(uTextureSpecular0, _TexCoord).rgb;
    vec3 NormalW = normalize(_NormalW);

    vec3 Diffuse = vec3(0.0f);
    vec3 Specular = vec3(0.0f);
    float DiffuseFactor = dot(NormalW, normalize(-uLightDirectionW));
    if (DiffuseFactor > 0.0f) {
        Diffuse = uLightDiffuse * DiffuseColor;

        vec3 Vertex2Eye = normalize(uEyePosW - _PositionW);
        vec3 ReflectLight = normalize(reflect(uLightDirectionW, normalize(NormalW)));
        float Specularvactor = dot(Vertex2Eye, ReflectLight);
        if (Specularvactor > 0.0f) {
            Specular = uLightSpecular * SpecularColor;
        }
    }

    _outFragColor = vec4(Diffuse + Specular, 1.0f);
}