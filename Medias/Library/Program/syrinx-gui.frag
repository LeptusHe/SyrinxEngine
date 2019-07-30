layout(location = 0) in vec2 inUV;
layout(location = 1) in vec4 inColor;

uniform sampler2D inTex;

layout(location = 0) out vec4 outColor;


void main()
{
    vec4 color = inColor / vec4(255.0);
    outColor = color * texture(inTex, inUV);
}