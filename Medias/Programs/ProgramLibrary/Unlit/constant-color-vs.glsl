uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;

layout (location = 0) in vec3 _inVertexPosition;

out gl_PerVertex {
    vec4 gl_Position;
};


void main()
{
    gl_Position = uProjectionMatrix * uViewMatrix * uModelMatrix * vec4(_inVertexPosition, 1.0);
}
