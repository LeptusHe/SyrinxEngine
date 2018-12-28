#version 450 core

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;

uniform mat4 uCurrentProjectionMatrix;
uniform mat4 uHistoryViewMatrix;
uniform mat4 uHistoryProjectionMatrix;

layout (location = 0) in vec3 _inVertexPosition;
layout (location = 1) in vec3 _inVertexNormal;
layout (location = 2) in vec2 _inVertexTexCoord;

out vec3 _NormalW;
out vec2 _TexCoords;
out vec3 _PositionW;
out vec4 _HistoryPositionC;
out vec4 _CurrentPositionC;

out gl_PerVertex
{
    vec4 gl_Position;
    float gl_PointSize;
    float gl_ClipDistance[];
};

vec3 getNDCCoord(vec4 positionC);

void main()
{
	//gl_Position = vec4(uProjectionMatrix * uViewMatrix * uModelMatrix * vec4(_inVertexPosition, 1.0));
	//if (getNDCCoord(gl_Position).x > 0.5) {
    //    gl_Position = vec4(uCurrentProjectionMatrix * uViewMatrix * uModelMatrix * vec4(_inVertexPosition, 1.0));
    //}
	_NormalW = normalize(transpose(inverse(mat3(uModelMatrix))) * _inVertexNormal);;
	_TexCoords = _inVertexTexCoord;
	_PositionW = vec4(uModelMatrix * vec4(_inVertexPosition, 1.0)).xyz;

	_CurrentPositionC = uCurrentProjectionMatrix * uViewMatrix * vec4(_PositionW, 1.0f);
	_HistoryPositionC = uHistoryProjectionMatrix * uHistoryViewMatrix * vec4(_PositionW, 1.0f);

	gl_Position = vec4(uCurrentProjectionMatrix * uViewMatrix * uModelMatrix * vec4(_inVertexPosition, 1.0));
	gl_Position = vec4(uProjectionMatrix * uViewMatrix * uModelMatrix * vec4(_inVertexPosition, 1.0));
}

vec3 getNDCCoord(vec4 positionC)
{
	vec3 PositionNDC = positionC.xyz / vec3(positionC.w);
	PositionNDC = (PositionNDC + vec3(1.0f)) / vec3(2.0f);
	return PositionNDC;
}