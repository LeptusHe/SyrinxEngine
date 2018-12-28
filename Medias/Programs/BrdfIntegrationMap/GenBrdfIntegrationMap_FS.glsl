#version 430 core

in vec2 _TexCoord;

out vec3 _outFragColor;

const float PI = 3.14159265359;

float radicalInverseVDC(uint bits);
vec2 hammersley(uint i, uint N);
vec3 importanceSampleGGX(vec2 Xi, vec3 N, float roughness);
float geometrySchlickGGXForIBL(float NdotV, float roughness);
float geometrySmithForIBL(vec3 N, vec3 V, vec3 L, float roughness);
vec2 integrateBRDF(float NdotV, float roughness);

void main() {
    vec2 IntegrateBRDF = integrateBRDF(_TexCoord.x, _TexCoord.y);
    IntegrateBRDF = pow(IntegrateBRDF.rg, vec2(1.0f / 2.2f));
    _outFragColor = vec3(IntegrateBRDF, 0.0f);
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

// ----------------------------------------------------------------------------
float geometrySchlickGGXForIBL(float NdotV, float roughness)
{
    float a = roughness;
    float k = (a * a) / 2.0f;
    float nom   = NdotV;
    float denom = NdotV * (1.0f - k) + k;

    return nom / denom;
}


float geometrySmithForIBL(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0f);
    float NdotL = max(dot(N, L), 0.0f);
    float ggx2 = geometrySchlickGGXForIBL(NdotV, roughness);
    float ggx1 = geometrySchlickGGXForIBL(NdotL, roughness);

    return ggx1 * ggx2;
}


vec2 integrateBRDF(float NdotV, float roughness)
{
    vec3 V = vec3(sqrt(1.0f - NdotV * NdotV), 0.0f, NdotV);
    float A = 0.0f;
    float B = 0.0f;
    vec3 N = vec3(0.0f, 0.0f, 1.0f);

    const uint SAMPLE_COUNT = 1024u;
    for(uint i = 0u; i < SAMPLE_COUNT; i++){
        vec2 Xi = hammersley(i, SAMPLE_COUNT);
        vec3 H = importanceSampleGGX(Xi, N, roughness);
        vec3 L = normalize(2.0f * dot(V, H) * H - V);

        float NdotL = max(L.z, 0.0f);
        float NdotH = max(H.z, 0.0f);
        float VdotH = max(dot(V, H), 0.0f);

        if(NdotL > 0.0f){
            float G = geometrySmithForIBL(N, V, L, roughness);
            float GVis = (G * VdotH) / (NdotH * NdotV);
            float Fc = pow(1.0f - VdotH, 5.0f);

            A += (1.0f - Fc) * GVis;
            B += Fc * GVis;
        }
    }

    A /= float(SAMPLE_COUNT);
    B /= float(SAMPLE_COUNT);
    return vec2(A, B);
}
