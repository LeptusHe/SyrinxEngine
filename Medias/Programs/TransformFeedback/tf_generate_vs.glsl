#version 450 core

layout (location = 0) in vec3 _inVertexPosition;

out vec3 VERTEX_OBJECT_POSITION;
out vec4 VERTEX_COLOR;

out gl_PerVertex
{
    vec4 gl_Position;
    float gl_PointSize;
    float gl_ClipDistance[];
};

void main()
{
	gl_Position = vec4(_inVertexPosition, 1.0f);
	VERTEX_OBJECT_POSITION = _inVertexPosition;
	VERTEX_COLOR = vec4(1.0f, 1.0f, 0.0f, 1.0f);
}