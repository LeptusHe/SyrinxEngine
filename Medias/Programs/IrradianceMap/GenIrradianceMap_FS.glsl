#version 430 core

uniform samplerCube uEnvironmentMap;

in vec3 _LocalPos;

out vec4 _outFragColor;

const float PI = 3.14159265359;

void main()
{
    vec3 Normal = normalize(_LocalPos);
    vec3 Irradiance = vec3(0.0f);
    vec3 Up = vec3(0.0f, 1.0f, 0.0f);
    vec3 Right = cross(Up, Normal);
    Up = cross(Normal, Right);

    float SampleDelta = 0.025f;
    int SamplesCount = 0;

    for(float Phi = 0.0f; Phi < 2.0f * PI; Phi += SampleDelta){
        for(float Theta = 0.0f; Theta < 0.5f * PI; Theta += SampleDelta){
            vec3 TangentSample = vec3(sin(Theta) * cos(Phi), sin(Theta) * sin(Phi), cos(Theta));
            vec3 SampleVector = TangentSample.x * Right + TangentSample.y * Up + TangentSample.z * Normal;
            Irradiance += texture(uEnvironmentMap, SampleVector).rgb * cos(Theta) * sin(Theta);
            SamplesCount++;
        }
    }

    Irradiance = PI * Irradiance * (1.0f / float(SamplesCount));
    Irradiance = pow(Irradiance.rgb, vec3(1.0f / 2.2f));
    _outFragColor = vec4(Irradiance, 1.0f);
}
