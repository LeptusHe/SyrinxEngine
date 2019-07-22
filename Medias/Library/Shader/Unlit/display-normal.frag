layout (location = 0) in vec3 inNormal;
layout (location = 0) out vec4 outColor;

void main()
{
    vec3 normal = inNormal * 0.5 + 0.5;
    outColor = vec4(normal, 1.0);
}