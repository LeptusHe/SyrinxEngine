#version 450 core

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;

layout (location = 0) in vec3 _inVertexPosition;
layout (location = 1) in vec3 _inVertexNormal;
layout (location = 2) in vec3 _inVertexTangent;
layout (location = 3) in vec3 _inVertexBitangent;
layout (location = 4) in vec2 _inVertexTexCoord;

out gl_PerVertex {
    vec4 gl_Position;
};


out vec2 _TexCoord;
out mat3 _TBN;

void main()
{
    _TexCoord = _inVertexTexCoord;

    vec3 T = normalize(vec3(uModelMatrix * vec4(_inVertexTangent, 0.0f)));
    vec3 B = normalize(vec3(uModelMatrix * vec4(_inVertexBitangent, 0.0f)));
    vec3 N = normalize(vec3(uModelMatrix * vec4(_inVertexNormal, 0.0f)));
    _TBN = mat3(T, B, N);

    gl_Position = uProjectionMatrix * uViewMatrix * uModelMatrix * vec4(_inVertexPosition, 1.0);
}
