#version 450 core
#include <PostProcessTemporalAA.prog>

uniform sampler2D uCurrentSceneSampler;
uniform sampler2D uHistorySceneSampler;
uniform sampler2D uMotionVectorSampler;
uniform sampler2D uCurrentSceneDepthSampler;

uniform float uUseNeighborhoodClamping = 0.0f;
uniform float uUseNeighborhoodCliping = 0.0f;
uniform float uEnableClosestFragment = 1.0f;

in vec2 _TexCoord;
layout (location = 0) out vec4 _outFragColor;
layout (location = 1) out vec4 _outSegmentationMask;


void main()
{
    vec2 motionTexCoord = _TexCoord;
    ivec2 velocityOffset = ivec2(0);
    vec2 dilatedMotionTexCoord = FindClosestFragment(uCurrentSceneDepthSampler, _TexCoord, velocityOffset).xy;
    if (uEnableClosestFragment > 0.5) {
        motionTexCoord = dilatedMotionTexCoord;
    }

    vec2 motion = texture(uMotionVectorSampler, motionTexCoord).xy;
    vec3 currentColor = texture(uCurrentSceneSampler, _TexCoord).rgb;
    vec3 historyColor = texture(uHistorySceneSampler, _TexCoord + motion).rgb;
    if (uUseNeighborhoodClamping > 0.5f || uUseNeighborhoodCliping > 0.5f) {
        historyColor = ClampHistoryColor(historyColor, currentColor, uCurrentSceneSampler, _TexCoord);
    }
    _outFragColor = vec4(mix(historyColor, currentColor, 0.05f), 1.0f);

    _outSegmentationMask = vec4(0.0f, 0.0f, 1.0f, 1.0f);
    if (all(equal(velocityOffset, ivec2(0.0)))) {
        if (uEnableClosestFragment > 0.5) {
            _outSegmentationMask = vec4(0.0, 1.0, 0.0, 1.0);
        } else {
            _outSegmentationMask = vec4(0.0, 0.0, 1.0, 1.0);
        }
    } else {
        _outSegmentationMask = vec4(1.0, 0.0, 0.0, 1.0);
    }
}