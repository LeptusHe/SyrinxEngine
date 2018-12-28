#version 450 core
#extension GL_ARB_bindless_texture : require
#extension GL_ARB_gpu_shader_int64 : enable

in vec3 _NormalW;
in vec2 _TexCoord;
in vec3 _PositionW;
in vec4 _Color;
flat in float _DiffuseTextureIndex;
flat in float _SpecularTextureIndex;
flat in float _Metallic;
flat in float _Roughness;
flat in float _UsePBR;

layout (location = 0) out vec4 _outPositionW;
layout (location = 1) out vec3 _outNormalW;
layout (location = 2) out vec4 _outAlbedoAndUsePBR;
layout (location = 3) out float _outMetallic;
layout (location = 4) out float _outRoughness;

struct Sampler{
    sampler2D texture;
    uint64_t padding;
};

layout (std140, binding = 0) uniform TextureHandles {
   Sampler allTheSamplers0[25];
};

void main()
{
	_outPositionW = vec4(_PositionW, gl_FragCoord.z);
	_outNormalW = normalize(_NormalW);

	vec3 Color = vec3(1.0f, 0.0f, 0.0f);

	if (_DiffuseTextureIndex < 99.0f) {
	    int Index = int(_DiffuseTextureIndex);
	    Color = texture(allTheSamplers0[Index].texture, _TexCoord).rgb;
	}
	_outAlbedoAndUsePBR = vec4(Color, _UsePBR);

    if (_UsePBR > 0.5f) {
        _outAlbedoAndUsePBR.rgb = vec3(0.827f, 0.659f, 0.353f);
    }

    _outMetallic  = _Metallic;
    _outRoughness = _Roughness;
}