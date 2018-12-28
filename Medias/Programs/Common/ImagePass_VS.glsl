#version 450 core

layout (location = 0) in vec3 _inVertexPosition;
layout (location = 1) in vec2 _inVertexTexCoord;

out vec2 _TexCoord;
out gl_PerVertex
{
    vec4 gl_Position;
    float gl_PointSize;
    float gl_ClipDistance[];
};

void main(void)
{
	gl_Position = vec4(_inVertexPosition.xy, 0.0f, 1.0f);
	_TexCoord = _inVertexTexCoord.xy;
}