#version 430 core

uniform samplerCube uEnvironmentMap;
uniform float uRoughness;
uniform float uResolution;

in vec3 _LocalPos;

out vec4 _outFragColor;

const float PI = 3.14159265359;

float radicalInverseVDC(uint bits);
vec2 hammersley(uint i, uint N);
vec3 importanceSampleGGX(vec2 Xi, vec3 N, float roughness);
float distributionGGX(vec3 N, vec3 H, float roughness);

void main()
{
    vec3 N = normalize(_LocalPos);
    vec3 R = N;
    vec3 V = N;

    const uint SAMPLE_COUNT = 1024u;
    float TotalWeight = 0.0f;
    vec3 PreFilteredColor = vec3(0.0f);
    for(uint i = 0u; i < SAMPLE_COUNT; i++){
        vec2 Xi = hammersley(i, SAMPLE_COUNT);
        vec3 H = importanceSampleGGX(Xi, N, uRoughness);
        vec3 L = normalize(2.0f * dot(V, H) * H - V);

        float NdotH = max(dot(N, H), 0.0f);
        float HdotV = max(dot(H, V), 0.0f);
        float NdotL = max(dot(N, L), 0.0f);
        if(NdotL > 0.0f){
            float D = distributionGGX(N, H, uRoughness);
            float PDF = (D * NdotH / (4.0f * HdotV)) + 0.0001f;
            float SaTexel = 4.0f * PI / (6.0f * uResolution * uResolution);
            float SaSample = 1.0f / (float(SAMPLE_COUNT) * PDF + 0.0001);
            float MipLevel = uRoughness == 0.0f ? 0.0f : 0.5f * log2(SaSample / SaTexel);
            PreFilteredColor += textureLod(uEnvironmentMap, L, MipLevel).rgb * NdotL;
            TotalWeight += NdotL;
        }
    }

    PreFilteredColor = PreFilteredColor / TotalWeight;
    PreFilteredColor = pow(PreFilteredColor.rgb, vec3(1.0f / 2.2f));
    _outFragColor = vec4(PreFilteredColor, 1.0f);
}


float radicalInverseVDC(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}


vec2 hammersley(uint i, uint N)
{
    return vec2(float(i)/float(N), radicalInverseVDC(i));
}


vec3 importanceSampleGGX(vec2 Xi, vec3 N, float roughness)
{
    float a = roughness * roughness;

    float Phi = 2.0f * PI * Xi.x;
    float CosTheta = sqrt((1.0f - Xi.y) / (1.0f + (a * a - 1.0f) * Xi.y));
    float SinTheta = sqrt(1.0f - CosTheta * CosTheta);

    // from spherical coordinates to cartesian coordinates
    vec3 H;
    H.x = cos(Phi) * SinTheta;
    H.y = sin(Phi) * SinTheta;
    H.z = CosTheta;

    // from tangent-space vector to world-space sample vector
    vec3 Up        = abs(N.z) < 0.999f ? vec3(0.0f, 0.0f, 1.0f) : vec3(1.0f, 0.0f, 0.0f);
    vec3 Tangent   = normalize(cross(Up, N));
    vec3 Bitangent = cross(N, Tangent);

    vec3 SampleVec = Tangent * H.x + Bitangent * H.y + N * H.z;
    return normalize(SampleVec);
}


float distributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0f);
    float NdotH2 = NdotH * NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = PI * denom * denom;

    return nom / denom;
}