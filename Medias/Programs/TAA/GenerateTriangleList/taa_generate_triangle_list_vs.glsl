#version 450 core

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;

uniform sampler2D uSegmentationMask;

uniform vec3 uBoundingBoxVertices[8];
uniform vec2 uViewPort;
uniform float uDiffuseTexIndex;
uniform float uSpecularTexIndex;

layout (location = 0) in vec3 _inVertexPosition;
layout (location = 1) in vec3 _inVertexNormal;
layout (location = 2) in vec2 _inVertexTexCoord;

flat out int _BoundingVisible;
out vec3 _VertexPosition;
out vec3 _Normal;
out vec2 _TexCoord;
out float _DiffuseTexIndex;
out float _SpecularTexIndex;

out gl_PerVertex
{
    vec4 gl_Position;
    float gl_PointSize;
    float gl_ClipDistance[];
};

const float MIN_VALUE = 0.000001f;

vec4 BoundingBox[8];

int instanceCloudReduction();
int boundingBoxJudgment();

void main()
{
    _VertexPosition = _inVertexPosition;
    _Normal = normalize(transpose(inverse(mat3(uModelMatrix))) * _inVertexNormal);
    _TexCoord = _inVertexTexCoord;
    _DiffuseTexIndex = uDiffuseTexIndex;
    _SpecularTexIndex = uSpecularTexIndex;
	gl_Position = vec4(uProjectionMatrix * uViewMatrix * uModelMatrix * vec4(_inVertexPosition, 1.0));
	_BoundingVisible = boundingBoxJudgment();
}


int instanceCloudReduction()
{
    mat4 MVPMatrix = uProjectionMatrix * uViewMatrix * uModelMatrix;
    for (int i = 0; i < 8; ++i) {
        BoundingBox[i] = MVPMatrix * vec4(uBoundingBoxVertices[i], 1.0f);
    }

    int OutOfBound[6] = int[6](0, 0, 0, 0, 0, 0);

    for (int i = 0; i < 8; ++i) {
        if ( BoundingBox[i].x >  BoundingBox[i].w ) OutOfBound[0]++;
        if ( BoundingBox[i].x < -BoundingBox[i].w ) OutOfBound[1]++;
        if ( BoundingBox[i].y >  BoundingBox[i].w ) OutOfBound[2]++;
        if ( BoundingBox[i].y < -BoundingBox[i].w ) OutOfBound[3]++;
        if ( BoundingBox[i].z >  BoundingBox[i].w ) OutOfBound[4]++;
        if ( BoundingBox[i].z < -BoundingBox[i].w ) OutOfBound[5]++;
    }

    int InFrustum = 1;

    for (int i = 0; i < 6; ++i) {
        if (OutOfBound[i] == 8) InFrustum = 0;
    }

    return InFrustum;
}


int boundingBoxJudgment()
{
    if (instanceCloudReduction() == 0) return 0;

    for (int i = 0; i < 8; ++i) {
        BoundingBox[i].xyz /= BoundingBox[i].w;
    }

    vec2 BoundingRect[2];
    vec2 BoundingRectC[2];

    BoundingRect[0].x = min( min( min( BoundingBox[0].x, BoundingBox[1].x ),
    							  min( BoundingBox[2].x, BoundingBox[3].x ) ),
    						 min( min( BoundingBox[4].x, BoundingBox[5].x ),
    							  min( BoundingBox[6].x, BoundingBox[7].x ) ) );
    BoundingRect[0].y = min( min( min( BoundingBox[0].y, BoundingBox[1].y ),
    							  min( BoundingBox[2].y, BoundingBox[3].y ) ),
    						 min( min( BoundingBox[4].y, BoundingBox[5].y ),
    							  min( BoundingBox[6].y, BoundingBox[7].y ) ) );
    BoundingRect[1].x = max( max( max( BoundingBox[0].x, BoundingBox[1].x ),
    							  max( BoundingBox[2].x, BoundingBox[3].x ) ),
    						 max( max( BoundingBox[4].x, BoundingBox[5].x ),
    							  max( BoundingBox[6].x, BoundingBox[7].x ) ) );
    BoundingRect[1].y = max( max( max( BoundingBox[0].y, BoundingBox[1].y ),
    							  max( BoundingBox[2].y, BoundingBox[3].y ) ),
    						 max( max( BoundingBox[4].y, BoundingBox[5].y ),
    							  max( BoundingBox[6].y, BoundingBox[7].y ) ) );

    BoundingRectC[0] = BoundingRect[0] / vec2(2.0f) + vec2(0.5f);
    BoundingRectC[1] = BoundingRect[1] / vec2(2.0f) + vec2(0.5f);

    float ViewSizeX = (BoundingRectC[1].x - BoundingRectC[0].x) * uViewPort.x;
    float ViewSizeY = (BoundingRectC[1].y - BoundingRectC[0].y) * uViewPort.y;

    float LOD = ceil(log2(max(ViewSizeX, ViewSizeY) / 2.0f));

    vec3 SampleA = textureLod(uSegmentationMask, vec2(BoundingRectC[0].x, BoundingRectC[0].y), LOD).rgb;
    vec3 SampleB = textureLod(uSegmentationMask, vec2(BoundingRectC[0].x, BoundingRectC[1].y), LOD).rgb;
    vec3 SampleC = textureLod(uSegmentationMask, vec2(BoundingRectC[1].x, BoundingRectC[1].y), LOD).rgb;
    vec3 SampleD = textureLod(uSegmentationMask, vec2(BoundingRectC[1].x, BoundingRectC[0].y), LOD).rgb;
    vec3 SampleE = textureLod(uSegmentationMask, vec2(BoundingRectC[0].x, (BoundingRectC[0].y + BoundingRectC[1].y) * 0.5f), LOD).rgb;
    vec3 SampleF = textureLod(uSegmentationMask, vec2((BoundingRectC[0].x + BoundingRectC[1].x) * 0.5f, BoundingRectC[1].y), LOD).rgb;
    vec3 SampleG = textureLod(uSegmentationMask, vec2(BoundingRectC[1].x, (BoundingRectC[0].y + BoundingRectC[1].y) * 0.5f), LOD).rgb;
    vec3 SampleH = textureLod(uSegmentationMask, vec2((BoundingRectC[0].x + BoundingRectC[1].x) * 0.5f, BoundingRectC[0].y), LOD).rgb;
    vec3 SampleI = textureLod(uSegmentationMask, vec2((BoundingRectC[0].x + BoundingRectC[1].x) * 0.5f, (BoundingRectC[0].y + BoundingRectC[1].y) * 0.5f), LOD).rgb;

    if (abs(SampleA.x - 0.0f) > MIN_VALUE || abs(SampleA.y - 0.0f) > MIN_VALUE || abs(SampleA.z - 1.0f) > MIN_VALUE) {
        return 1;
    }
    if (abs(SampleB.x - 0.0f) > MIN_VALUE || abs(SampleB.y - 0.0f) > MIN_VALUE || abs(SampleB.z - 1.0f) > MIN_VALUE) {
        return 1;
    }
    if (abs(SampleC.x - 0.0f) > MIN_VALUE || abs(SampleC.y - 0.0f) > MIN_VALUE || abs(SampleC.z - 1.0f) > MIN_VALUE) {
        return 1;
    }
    if (abs(SampleD.x - 0.0f) > MIN_VALUE || abs(SampleD.y - 0.0f) > MIN_VALUE || abs(SampleD.z - 1.0f) > MIN_VALUE) {
        return 1;
    }
    if (abs(SampleE.x - 0.0f) > MIN_VALUE || abs(SampleE.y - 0.0f) > MIN_VALUE || abs(SampleE.z - 1.0f) > MIN_VALUE) {
        return 1;
    }
    if (abs(SampleF.x - 0.0f) > MIN_VALUE || abs(SampleF.y - 0.0f) > MIN_VALUE || abs(SampleF.z - 1.0f) > MIN_VALUE) {
        return 1;
    }
    if (abs(SampleG.x - 0.0f) > MIN_VALUE || abs(SampleG.y - 0.0f) > MIN_VALUE || abs(SampleG.z - 1.0f) > MIN_VALUE) {
        return 1;
    }
    if (abs(SampleH.x - 0.0f) > MIN_VALUE || abs(SampleH.y - 0.0f) > MIN_VALUE || abs(SampleH.z - 1.0f) > MIN_VALUE) {
        return 1;
    }
    if (abs(SampleI.x - 0.0f) > MIN_VALUE || abs(SampleI.y - 0.0f) > MIN_VALUE || abs(SampleI.z - 1.0f) > MIN_VALUE) {
        return 1;
    }

    return 0;
}