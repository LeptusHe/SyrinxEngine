#version 450 core
#extension GL_ARB_bindless_texture : require
#extension GL_ARB_gpu_shader_int64 : enable

in vec2 _TexCoord;
in vec4 _Color;
flat in float _DiffuseTextureIndex;
flat in float _SpecularTextureIndex;

struct Sampler{
    sampler2D texture;
    uint64_t padding;
};

layout (std140, binding = 0) uniform TextureHandles {
   Sampler allTheSamplers0[25];
};

out vec4 _outFragColor;

void main()
{
    _outFragColor = _Color;

//    if (_DiffuseTextureIndex < 99.0f) {
//        int Index = int(_DiffuseTextureIndex);
//        _outFragColor.rgb = texture(allTheSamplers0[Index].texture, _TexCoord).rgb;
//    }
}