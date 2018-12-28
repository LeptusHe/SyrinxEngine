#version 430 core

uniform samplerCube uCubeMap;

in vec3 _TexCoord;

out vec4 _outFragColor;

void main()
{
	_outFragColor = texture(uCubeMap, _TexCoord);
}