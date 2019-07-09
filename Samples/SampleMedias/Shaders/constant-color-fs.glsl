layout(std140, binding = 1) uniform ColorBuffer {
    vec3 displayColor;
};

layout(location = 0) out vec4 _outFragColor;

void main()
{
    _outFragColor = vec4(displayColor, 1.0);
}