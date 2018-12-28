#version 450 core

uniform sampler2D uTextureDiffuse0;

uniform float uMetallic = 0.0f;
uniform float uRoughness = 0.0f;
uniform float uUsePBR = 0.0f;

in vec3 _NormalW;
in vec2 _TexCoords;
in vec3 _PositionW;

layout (location = 0) out vec4 _outPositionW;
layout (location = 1) out vec3 _outNormalW;
layout (location = 2) out vec4 _outAlbedoAndUsePBR;
layout (location = 3) out float _outMetallic;
layout (location = 4) out float _outRoughness;

void main()
{
	_outPositionW = vec4(_PositionW, gl_FragCoord.z);
	_outNormalW = normalize(_NormalW);
	_outAlbedoAndUsePBR = vec4(texture2D(uTextureDiffuse0, _TexCoords).rgb, uUsePBR);
    if (uUsePBR > 0.5f) {
        _outAlbedoAndUsePBR.rgb = vec3(0.827f, 0.659f, 0.353f);
    }
    _outMetallic  = uMetallic;
    _outRoughness = uRoughness;
}