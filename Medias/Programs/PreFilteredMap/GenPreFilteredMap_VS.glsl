#version 430 core

uniform mat4 uProjectionMatrix;
uniform mat4 uViewMatrix;

layout (location = 0) in vec3 _inVertexPosition;

out vec3 _LocalPos;
out gl_PerVertex
{
    vec4 gl_Position;
    float gl_PointSize;
    float gl_ClipDistance[];
};

void main()
{
    _LocalPos = _inVertexPosition;

    vec4 ClipPos = uProjectionMatrix * mat4(mat3(uViewMatrix)) * vec4(_inVertexPosition, 1.0f);
    gl_Position = ClipPos.xyww;
}