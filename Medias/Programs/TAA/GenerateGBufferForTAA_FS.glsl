#version 450 core

uniform sampler2D uTextureDiffuse0;

uniform float uMetallic = 0.0f;
uniform float uRoughness = 0.0f;
uniform float uUsePBR = 0.0f;
uniform float uMipmapBias = 0.0f;

in vec3 _NormalW;
in vec2 _TexCoords;
in vec3 _PositionW;
in vec4 _HistoryPositionC;
in vec4 _CurrentPositionC;

layout (location = 0) out vec4 _outPositionW;
layout (location = 1) out vec3 _outNormalW;
layout (location = 2) out vec4 _outAlbedoAndUsePBR;
layout (location = 3) out float _outMetallic;
layout (location = 4) out float _outRoughness;
layout (location = 5) out vec2 _outMotionVector;

vec3 getNDCCoord(vec4 vPositionC);
vec2 calculateMotionVector(vec4 vHistoryPositionC, vec4 vCurrentPositionC);

void main()
{
	_outPositionW = vec4(_PositionW, gl_FragCoord.z);
	_outNormalW = normalize(_NormalW);
	_outAlbedoAndUsePBR = vec4(texture(uTextureDiffuse0, _TexCoords, uMipmapBias).rgb, uUsePBR);

	if (uUsePBR > 0.5f) {
	    _outAlbedoAndUsePBR.rgb = vec3(0.827f, 0.659f, 0.353f);
	    //_outAlbedoAndUsePBR.rgb = vec3(118.0f / 256.0f, 119.0f / 256.0f, 120.0f / 256.0f);
	}
    _outMetallic  = uMetallic;
    _outRoughness = uRoughness;
	vec2 MotionVector = calculateMotionVector(_HistoryPositionC, _CurrentPositionC);
	_outMotionVector = MotionVector;
}


vec3 getNDCCoord(vec4 positionC)
{
	vec3 PositionNDC = positionC.xyz / vec3(positionC.w);
	PositionNDC = (PositionNDC + vec3(1.0f)) / vec3(2.0f);
	return PositionNDC;
}


vec2 calculateMotionVector(vec4 historyPositionC, vec4 currentPositionC)
{
	vec2 HistoryPosNDC = getNDCCoord(historyPositionC).xy;
	vec2 CurrentPosNDC = getNDCCoord(currentPositionC).xy;

	vec2 MotionVector = (HistoryPosNDC - CurrentPosNDC);
	return MotionVector;
}