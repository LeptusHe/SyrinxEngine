#version 450 core

layout (triangles) in;
layout (triangle_strip, max_vertices = 9) out;

flat in int _Visible[];
in vec4 _BoundingRect[][4];

in gl_PerVertex
{
    vec4  gl_Position;
    float gl_PointSize;
    float gl_ClipDistance[];
} gl_in[];

out int _VisibleG;
out gl_PerVertex
{
    vec4 gl_Position;
    float gl_PointSize;
    float gl_ClipDistance[];
};

void main() {
    for (int i = 0; i < gl_in.length(); ++i) {
        gl_Position = gl_in[i].gl_Position;
        _VisibleG = _Visible[0];
        EmitVertex();
    }
    EndPrimitive();

    gl_Position = _BoundingRect[0][1];
    _VisibleG = 2;
    EmitVertex();
    gl_Position = _BoundingRect[0][0];
    _VisibleG = 2;
    EmitVertex();
    gl_Position = _BoundingRect[0][2];
    _VisibleG = 2;
    EmitVertex();
    EndPrimitive();

    gl_Position = _BoundingRect[0][2];
    _VisibleG = 2;
    EmitVertex();
    gl_Position = _BoundingRect[0][0];
    _VisibleG = 2;
    EmitVertex();
    gl_Position = _BoundingRect[0][3];
    _VisibleG = 2;
    EmitVertex();
    EndPrimitive();
}
