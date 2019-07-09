layout(std140, binding = 0) uniform Buffer {
    float displayColor;
};

layout(location = 0) out vec4 _outFragColor;

void main()
{
    _outFragColor = vec4(displayColor);
}