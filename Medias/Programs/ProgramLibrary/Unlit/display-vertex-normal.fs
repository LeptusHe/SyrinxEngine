#version 450 core

uniform vec4 displayColor;
out vec4 _outFragColor;

in vec3 _normal;

void main()
{
    vec3 normal = (_normal + vec3(1.0)) * vec3(0.5);
    _outFragColor = vec4(normal, 1.0);
}
