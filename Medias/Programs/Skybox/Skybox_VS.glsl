#version 430 core

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;

layout (location = 0) in vec3 _inVertexPosition;

out vec3 _TexCoord;
out gl_PerVertex
{
    vec4 gl_Position;
    float gl_PointSize;
    float gl_ClipDistance[];
};

void main()
{
	vec4 Position = uProjectionMatrix * mat4(mat3(uViewMatrix)) * uModelMatrix * vec4(_inVertexPosition, 1.0);
	_TexCoord = _inVertexPosition;
	gl_Position = Position.xyww;
}