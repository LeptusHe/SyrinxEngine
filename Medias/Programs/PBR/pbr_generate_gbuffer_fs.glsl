#version 450 core

uniform vec4 uAlbedo;
uniform float uMetallic;
uniform float uRoughness;

in vec3 _PositionW;
in vec3 _NormalW;
in vec2 _TexCoord;

layout (location = 0) out vec3 _outPositionW;
layout (location = 1) out vec3 _outNormalW;
layout (location = 2) out vec3 _outAlbedo;
layout (location = 3) out float _outMetallic;
layout (location = 4) out float _outRoughness;

void main()
{
    _outPositionW = _PositionW;
    _outNormalW   = _NormalW;
    _outAlbedo    = uAlbedo.rgb;
    _outMetallic  = uMetallic;
    _outRoughness = uRoughness;
}