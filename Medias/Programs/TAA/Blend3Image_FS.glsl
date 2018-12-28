#version 450 core

uniform sampler2D uSceneColorTex;
uniform sampler2D uSegmentationMask;

in float _MipMapLevel;
in vec2 _TexCoord;

out vec4 _outFragColor;

const float MIN_VALUE = 0.00001f;

void main()
{
    vec3 SegmentationMaskColorWithMipMap = textureLod(uSegmentationMask, _TexCoord, _MipMapLevel).rgb;
    if (abs(SegmentationMaskColorWithMipMap.r - 0.0f) > MIN_VALUE || abs(SegmentationMaskColorWithMipMap.g - 0.0f) > MIN_VALUE || abs(SegmentationMaskColorWithMipMap.b - 1.0f) > MIN_VALUE) {
        SegmentationMaskColorWithMipMap = vec3(0.2f, 0.2f, 0.2f);
    }
    vec3 SegmentationMaskColor = textureLod(uSegmentationMask, _TexCoord, 0).rgb;
    vec3 SceneColor = textureLod(uSceneColorTex, _TexCoord, 0).rgb;

	_outFragColor = vec4(SegmentationMaskColorWithMipMap, 1.0f);

    if (abs(SegmentationMaskColor.r - 1.0f) < MIN_VALUE && abs(SegmentationMaskColor.g - 1.0f) < MIN_VALUE && abs(SegmentationMaskColor.b - 0.0f) < MIN_VALUE) {
        _outFragColor = vec4(SegmentationMaskColor, 1.0f);
    } else if (abs(SceneColor.r - 1.0f) < MIN_VALUE && abs(SceneColor.g - 1.0f) < MIN_VALUE && abs(SceneColor.b - 0.0f) < MIN_VALUE){
	    _outFragColor = vec4(1.0f, 1.0f, 0.0f, 1.0f);
	} else if (abs(SceneColor.r - 0.0f) < MIN_VALUE && abs(SceneColor.g - 0.0f) < MIN_VALUE && abs(SceneColor.b - 0.0f) < MIN_VALUE){
        _outFragColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    } else if (abs(SceneColor.r - 1.0f) < MIN_VALUE && abs(SceneColor.g - 0.0f) < MIN_VALUE && abs(SceneColor.b - 0.0f) < MIN_VALUE){
        _outFragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
    }


    _outFragColor.rgb = SceneColor.rgb;
    if (abs(SegmentationMaskColor.r - 0.0f) > MIN_VALUE || abs(SegmentationMaskColor.g - 0.0f) > MIN_VALUE || abs(SegmentationMaskColor.b - 1.0f) > MIN_VALUE) {
        _outFragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
    }
}