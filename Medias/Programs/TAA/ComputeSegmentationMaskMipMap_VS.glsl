#version 450 core

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;

uniform vec3 uBoundingBoxVertices[8];
uniform vec3 uTriangleVertices[3];
uniform vec2 uViewPort;
uniform float uIsTriangle;
uniform float uShowSegmentationMaskMipmap = 1.0f;

layout (location = 0) in vec3 _inVertexPosition;
layout (location = 1) in vec2 _inVertexTexCoord;

out vec2 _TexCoord;
out float _MipMapLevel;
out gl_PerVertex
{
    vec4 gl_Position;
    float gl_PointSize;
    float gl_ClipDistance[];
};

float computeMipMapLevel4BoundingBox();
float computeMipMapLevel4Triangle();

void main(void)
{
	gl_Position = vec4(_inVertexPosition.xy, 0.0f, 1.0f);
	_TexCoord = _inVertexTexCoord.xy;
	_MipMapLevel = 0.0f;

    if (uShowSegmentationMaskMipmap > 0.5f) {
    	if (uIsTriangle < 0.5f) {
            _MipMapLevel = computeMipMapLevel4BoundingBox();
    	}
    	else {
    	    _MipMapLevel = computeMipMapLevel4Triangle();
    	}
    }
}

float computeMipMapLevel4BoundingBox()
{
    vec4 BoundingBox[8];
    mat4 MVPMatrix = uProjectionMatrix * uViewMatrix * uModelMatrix;
    for (int i = 0; i < 8; ++i) {
        BoundingBox[i] = MVPMatrix * vec4(uBoundingBoxVertices[i], 1.0f);
        BoundingBox[i].xyz /= BoundingBox[i].w;
    }
    vec2 BoundingRect[2];

    BoundingRect[0].x = min( min( min( BoundingBox[0].x, BoundingBox[1].x ),
    							  min( BoundingBox[2].x, BoundingBox[3].x ) ),
    						 min( min( BoundingBox[4].x, BoundingBox[5].x ),
    							  min( BoundingBox[6].x, BoundingBox[7].x ) ) ) / 2.0f + 0.5f;
    BoundingRect[0].y = min( min( min( BoundingBox[0].y, BoundingBox[1].y ),
    							  min( BoundingBox[2].y, BoundingBox[3].y ) ),
    						 min( min( BoundingBox[4].y, BoundingBox[5].y ),
    							  min( BoundingBox[6].y, BoundingBox[7].y ) ) ) / 2.0f + 0.5f;
    BoundingRect[1].x = max( max( max( BoundingBox[0].x, BoundingBox[1].x ),
    							  max( BoundingBox[2].x, BoundingBox[3].x ) ),
    						 max( max( BoundingBox[4].x, BoundingBox[5].x ),
    							  max( BoundingBox[6].x, BoundingBox[7].x ) ) ) / 2.0f + 0.5f;
    BoundingRect[1].y = max( max( max( BoundingBox[0].y, BoundingBox[1].y ),
    							  max( BoundingBox[2].y, BoundingBox[3].y ) ),
    						 max( max( BoundingBox[4].y, BoundingBox[5].y ),
    							  max( BoundingBox[6].y, BoundingBox[7].y ) ) ) / 2.0f + 0.5f;

    float ViewSizeX = (BoundingRect[1].x - BoundingRect[0].x) * uViewPort.x;
    float ViewSizeY = (BoundingRect[1].y - BoundingRect[0].y) * uViewPort.y;

    return ceil(log2(max(ViewSizeX, ViewSizeY) / 2.0f));
}


float computeMipMapLevel4Triangle()
{
    vec4 Triangle[3];
    mat4 MVPMatrix = uProjectionMatrix * uViewMatrix * uModelMatrix;
    for (int i = 0; i < 3; ++i) {
        Triangle[i] = MVPMatrix * vec4(uTriangleVertices[i], 1.0f);
        Triangle[i].xyz /= Triangle[i].w;
    }
    vec2 BoundingRect[2];

    BoundingRect[0].x = min(min(Triangle[0].x, Triangle[1].x), Triangle[2].x) / 2.0f + 0.5f;
    BoundingRect[0].y = min(min(Triangle[0].y, Triangle[1].y), Triangle[2].y) / 2.0f + 0.5f;
    BoundingRect[1].x = max(max(Triangle[0].x, Triangle[1].x), Triangle[2].x) / 2.0f + 0.5f;
    BoundingRect[1].y = max(max(Triangle[0].y, Triangle[1].y), Triangle[2].y) / 2.0f + 0.5f;

    float ViewSizeX = (BoundingRect[1].x - BoundingRect[0].x) * uViewPort.x;
    float ViewSizeY = (BoundingRect[1].y - BoundingRect[0].y) * uViewPort.y;

    return ceil(log2(max(ViewSizeX, ViewSizeY) / 2.0f));
}