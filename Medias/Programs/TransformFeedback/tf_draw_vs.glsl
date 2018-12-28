#version 450 core

layout (location = 0) in vec3 _inVertexPosition;
layout (location = 1) in vec4 _inVertexColor;

out vec4 _VertexColor;

out gl_PerVertex
{
    vec4 gl_Position;
    float gl_PointSize;
    float gl_ClipDistance[];
};

void main()
{
	gl_Position = vec4(_inVertexPosition.x + 0.1f, _inVertexPosition.y, _inVertexPosition.z, 1.0f);
	_VertexColor = _inVertexColor;
}