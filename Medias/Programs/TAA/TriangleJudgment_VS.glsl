#version 450 core

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;

uniform sampler2D uSegmentationMask;

uniform vec3 uTriangleVertices[3];
uniform vec2 uViewPort;

layout (location = 0) in vec3 _inVertexPosition;

flat out int _Visible;
out vec4 _BoundingRect[4];

out gl_PerVertex
{
    vec4 gl_Position;
    float gl_PointSize;
    float gl_ClipDistance[];
};

const float MIX_VALUE = 0.0001f;

vec4 Triangle[3];

int instanceCloudReduction();
int triangleJudgment();

void main()
{
	gl_Position = vec4(uProjectionMatrix * uViewMatrix * uModelMatrix * vec4(_inVertexPosition, 1.0f));
	_Visible = triangleJudgment();
}


int instanceCloudReduction()
{
    mat4 MVPMatrix = uProjectionMatrix * uViewMatrix * uModelMatrix;
    for (int i = 0; i < 3; ++i) {
        Triangle[i] = MVPMatrix * vec4(uTriangleVertices[i], 1.0f);
    }

    int OutOfBound[6] = int[6](0, 0, 0, 0, 0, 0);

    for (int i = 0; i < 3; ++i) {
         if ( Triangle[i].x >  Triangle[i].w ) OutOfBound[0]++;
         if ( Triangle[i].x < -Triangle[i].w ) OutOfBound[1]++;
         if ( Triangle[i].y >  Triangle[i].w ) OutOfBound[2]++;
         if ( Triangle[i].y < -Triangle[i].w ) OutOfBound[3]++;
         if ( Triangle[i].z >  Triangle[i].w ) OutOfBound[4]++;
         if ( Triangle[i].z < -Triangle[i].w ) OutOfBound[5]++;
    }

    int InFrustum = 1;

    for (int i = 0; i < 6; ++i) {
        if (OutOfBound[i] == 3) InFrustum = 0;
    }

    return InFrustum;
}


int triangleJudgment()
{
    if (instanceCloudReduction() == 0) return 0;

    for (int i = 0; i < 3; ++i) {
        Triangle[i].xyz /= Triangle[i].w;
    }

    vec2 BoundingRect[2];
    vec2 BoundingRectC[2];

    BoundingRect[0].x = min(min(Triangle[0].x, Triangle[1].x), Triangle[2].x);
    BoundingRect[0].y = min(min(Triangle[0].y, Triangle[1].y), Triangle[2].y);
    BoundingRect[1].x = max(max(Triangle[0].x, Triangle[1].x), Triangle[2].x);
    BoundingRect[1].y = max(max(Triangle[0].y, Triangle[1].y), Triangle[2].y);

    _BoundingRect[0] = vec4(BoundingRect[0].x, BoundingRect[1].y, 0.0f, 1.0f);
    _BoundingRect[1] = vec4(BoundingRect[1], 0.0f, 1.0f);
    _BoundingRect[2] = vec4(BoundingRect[1].x, BoundingRect[0].y, 0.0f, 1.0f);
    _BoundingRect[3] = vec4(BoundingRect[0], 0.0f, 1.0f);

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

    if (abs(SampleA.x - 0.0f) > MIX_VALUE || abs(SampleA.y - 0.0f) > MIX_VALUE || abs(SampleA.z - 1.0f) > MIX_VALUE) {
        return 1;
    }
    if (abs(SampleB.x - 0.0f) > MIX_VALUE || abs(SampleB.y - 0.0f) > MIX_VALUE || abs(SampleB.z - 1.0f) > MIX_VALUE) {
        return 1;
    }
    if (abs(SampleC.x - 0.0f) > MIX_VALUE || abs(SampleC.y - 0.0f) > MIX_VALUE || abs(SampleC.z - 1.0f) > MIX_VALUE) {
        return 1;
    }
    if (abs(SampleD.x - 0.0f) > MIX_VALUE || abs(SampleD.y - 0.0f) > MIX_VALUE || abs(SampleD.z - 1.0f) > MIX_VALUE) {
        return 1;
    }
    if (abs(SampleE.x - 0.0f) > MIX_VALUE || abs(SampleE.y - 0.0f) > MIX_VALUE || abs(SampleE.z - 1.0f) > MIX_VALUE) {
        return 1;
    }
    if (abs(SampleF.x - 0.0f) > MIX_VALUE || abs(SampleF.y - 0.0f) > MIX_VALUE || abs(SampleF.z - 1.0f) > MIX_VALUE) {
        return 1;
    }
    if (abs(SampleG.x - 0.0f) > MIX_VALUE || abs(SampleG.y - 0.0f) > MIX_VALUE || abs(SampleG.z - 1.0f) > MIX_VALUE) {
        return 1;
    }
    if (abs(SampleH.x - 0.0f) > MIX_VALUE || abs(SampleH.y - 0.0f) > MIX_VALUE || abs(SampleH.z - 1.0f) > MIX_VALUE) {
        return 1;
    }
    if (abs(SampleI.x - 0.0f) > MIX_VALUE || abs(SampleI.y - 0.0f) > MIX_VALUE || abs(SampleI.z - 1.0f) > MIX_VALUE) {
        return 1;
    }

    return 0;
}