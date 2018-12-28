#version 450 core

uniform sampler2D uTexSampler;
uniform int uMipMapLevel;

in vec2 _TexCoord;

out vec4 _outFragColor;

void main()
{
	_outFragColor = vec4(textureLod(uTexSampler, _TexCoord, uMipMapLevel).rgb, 1.0f);
}