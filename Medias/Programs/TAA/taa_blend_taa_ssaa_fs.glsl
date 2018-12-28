#version 450 core

uniform sampler2D uSceneColorForTAATex;
uniform sampler2D uSceneColorForSSAATex;
uniform sampler2D uSegmentationMask;
uniform float uUseImprovedTAA = 1.0f;

const float MIN_VALUE = 0.0001f;

in vec2 _TexCoord;
out vec4 _outFragColor;


void main()
{
    vec3 SegmentationMaskColor = textureLod(uSegmentationMask, _TexCoord, 0).rgb;
    vec3 SceneColorForSSAA = textureLod(uSceneColorForSSAATex, _TexCoord, 0).rgb;
    vec3 SceneColorForTAA = textureLod(uSceneColorForTAATex, _TexCoord, 0).rgb;

    _outFragColor = vec4(SceneColorForTAA, 1.0f);

    if (uUseImprovedTAA > 0.5) {
        if (abs(SegmentationMaskColor.r - 0.0f) > MIN_VALUE || abs(SegmentationMaskColor.g - 0.0f) > MIN_VALUE || abs(SegmentationMaskColor.b - 1.0f) > MIN_VALUE) {
            _outFragColor.rgb = SceneColorForSSAA;
        }
    }
}