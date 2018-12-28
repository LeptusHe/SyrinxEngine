#version 450 core

uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;

layout (location = 0) in vec4 _inVertexPositionW;
layout (location = 1) in vec3 _inVertexNormalW;
layout (location = 2) in vec4 _inVertexColor;
layout (location = 3) in vec2 _inVertexTexCoord;
layout (location = 4) in float _inDiffuseTextureIndex;
layout (location = 5) in float _inSpecularTextureIndex;
layout (location = 6) in float _inMetallic;
layout (location = 7) in float _inRoughness;
layout (location = 8) in float _inUsePBR;

out vec3 _NormalW;
out vec2 _TexCoord;
out vec3 _PositionW;
out vec4 _Color;
flat out float _DiffuseTextureIndex;
flat out float _SpecularTextureIndex;
flat out float _Metallic;
flat out float _Roughness;
flat out float _UsePBR;

out gl_PerVertex
{
    vec4 gl_Position;
    float gl_PointSize;
    float gl_ClipDistance[];
};

void main()
{
	gl_Position = uProjectionMatrix * uViewMatrix * _inVertexPositionW;
	_NormalW = _inVertexNormalW;
	_Color = _inVertexColor;
	_TexCoord = _inVertexTexCoord;
	_PositionW = _inVertexPositionW.xyz;
	_DiffuseTextureIndex = _inDiffuseTextureIndex;
    _SpecularTextureIndex = _inSpecularTextureIndex;
    _Metallic = _inMetallic;
    _Roughness = _inRoughness;
    _UsePBR = _inUsePBR;
}