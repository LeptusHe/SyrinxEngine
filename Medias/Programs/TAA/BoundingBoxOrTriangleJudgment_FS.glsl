#version 450 core

flat in int _VisibleG;

out vec4 _outFragColor;

void main()
{
    _outFragColor = vec4(0.0f, 0.0f, 1.0f, 1.0f);
    if (_VisibleG == 1) {
        _outFragColor = vec4(1.0f, 1.0f, 0.0f, 1.0f);
    }
    if (_VisibleG > 1) {
        _outFragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
    }

}