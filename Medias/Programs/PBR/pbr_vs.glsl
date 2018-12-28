#version 430 core

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;

layout (location = 0) in vec3 _inVertexPosition;
layout (location = 1) in vec3 _inVertexNormal;
layout (location = 2) in vec2 _inVertexTexCoord;

out vec3 _NormalW;
out vec2 _TexCoord;
out vec3 _PositionW;
out gl_PerVertex
{
    vec4 gl_Position;
    float gl_PointSize;
    float gl_ClipDistance[];
};

void main()
{
    _NormalW = normalize(transpose(inverse(mat3(uModelMatrix))) * _inVertexNormal);
    _TexCoord = _inVertexTexCoord;
    _PositionW = vec3(uModelMatrix * vec4(_inVertexPosition, 1.0));
    gl_Position = uProjectionMatrix * uViewMatrix * uModelMatrix * vec4(_inVertexPosition, 1.0f);
}