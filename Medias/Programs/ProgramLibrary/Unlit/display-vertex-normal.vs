#version 450 core

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;

layout (location = 0) in vec3 _inVertexPosition;
layout (location = 1) in vec3 _inVertexNormal;

out gl_PerVertex {
    vec4 gl_Position;
};


out vec3 _normal;

void main()
{
    gl_Position = uProjectionMatrix * uViewMatrix * uModelMatrix * vec4(_inVertexPosition, 1.0);
    _normal = transpose(inverse(mat3(uModelMatrix))) * _inVertexNormal;
}
