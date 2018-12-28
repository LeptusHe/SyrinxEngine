#version 450 core

in vec4 _VertexColor;

out vec4 _outFragColor;

void main()
{
	_outFragColor = _VertexColor;
}