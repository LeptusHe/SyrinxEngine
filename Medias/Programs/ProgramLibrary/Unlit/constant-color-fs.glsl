#version 450 core

uniform vec4 displayColor;
out vec4 _outFragColor;

void main()
{
    _outFragColor = vec4(displayColor);
}