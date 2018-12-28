#version 430 core

uniform float uAO;
uniform float uUseIBL = 0.0f;
uniform vec3 uCameraPos;
uniform vec3 uLightPositions[4];
uniform vec3 uLightColors[4];

// uniforms used by PBR Rendering
uniform sampler2D uGBufferPosition;
uniform sampler2D uGBufferNormal;
uniform sampler2D uGBufferAlbedoAndUsePBR;
uniform sampler2D uGBufferMetallic;
uniform sampler2D uGBufferRoughness;

// uniforms used by IBL
uniform sampler2D uBrdfIntegrationMap;
uniform samplerCube uIrradianceMap;
uniform samplerCube uPrefilteredMap;

// uniforms used by Simple light shading
uniform float uShininess = 64.0f;
uniform float uSpecularIntensity = 0.1f;

struct SBaseLight
{
    vec4 Color;
	float AmbientIntensity;
	float DiffuseIntensity;
	vec2 BaseLightAlign;
};
const float PI = 3.14159265359;

in vec2 _TexCoord;

out vec4 _outFragColor;

float distributionGGX(vec3 N, vec3 H, float roughness);
float geometrySchlickGGX(float NdotV, float roughness);
float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness);
vec3 fresnelSchlick(float cosTheta, vec3 F0);
vec3 fresnelSchlickRougness(float cosTheta, vec3 F0, float roughness);
vec3 shadingWithUnrealModelForImageBasedLight(vec3 albedo, float metallic, float roughness, vec3 normalW, vec3 viewDirW);
vec3 shadingWithUnrealModelForPointLight(vec3 albedo, float metallic, float roughness, vec3 normalW, vec3 positionW, vec3 lightColor[4], vec3 viewDirW, vec3 lightPosition[4]);
vec3 calcLightInternal(SBaseLight baseLight, vec3 lightDirection,  vec3 positionW, vec3 normal);

void main()
{
    vec3 PositionW = texture(uGBufferPosition, _TexCoord).rgb;
    vec3 NormalW = normalize(texture(uGBufferNormal, _TexCoord).rgb);
    vec4 AlbedoAndUsePBR = texture(uGBufferAlbedoAndUsePBR, _TexCoord).rgba;
    vec3 Albedo = AlbedoAndUsePBR.rgb;
    float UsePBR = AlbedoAndUsePBR.a;
    float Metallic = texture(uGBufferMetallic, _TexCoord).r;
    float Roughness = texture(uGBufferRoughness, _TexCoord).r;

    vec3 ViewDirection = normalize(uCameraPos - PositionW);

    vec3 Color = vec3(0.0f);

    if (UsePBR < 0.5f) {
        SBaseLight DirectionalLight;
        DirectionalLight.Color = vec4(1.0f);
        DirectionalLight.AmbientIntensity = 0.6f;
        DirectionalLight.DiffuseIntensity = 0.9f;
        vec3 DirectionalLightColor = calcLightInternal(DirectionalLight, vec3(-1.0, -1.0, 0.0), PositionW, NormalW);
        Color = Albedo * DirectionalLightColor;
    }  else {
        vec3 Lo = vec3(0.0f);
        if(uUseIBL < 0.5f){
            Lo = shadingWithUnrealModelForPointLight(Albedo, Metallic, Roughness, NormalW, PositionW, uLightColors, ViewDirection, uLightPositions);
        }else{
            Lo = shadingWithUnrealModelForImageBasedLight(Albedo, Metallic, Roughness, NormalW, ViewDirection);
        }
        vec3 Ambient = vec3(0.1f) * Albedo * uAO;
        Color = Ambient + Lo * 5.0;
        Color = Color / (Color + vec3(1.0f));
        Color = pow(Color, vec3(1.0f / 2.2f));
    }

    _outFragColor = vec4(Color, 1.0f);
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


float geometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0f);
    float k = (r * r) / 8.0f;

    float nom   = NdotV;
    float denom = NdotV * (1.0f - k) + k;

    return nom / denom;
}


float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0f);
    float NdotL = max(dot(N, L), 0.0f);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}


vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
}


vec3 fresnelSchlickRougness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0f - cosTheta, 5.0f);
}


vec3 shadingWithUnrealModelForPointLight(vec3 albedo, float metallic, float roughness, vec3 normalW, vec3 positionW, vec3 lightColor[4], vec3 viewDirW, vec3 lightPosition[4])
{
    const vec3 F0 = mix(vec3(0.04f), albedo, metallic);
    vec3 Lo = vec3(0.0f);

    for(int i = 0; i < 4; i++)
    {
        vec3 LightDirection = normalize(lightPosition[i] - positionW);
        vec3 HalfVector = normalize(viewDirW + LightDirection);
        float Distance = length(lightPosition[i] - positionW);
        float Attenuation = 1.0f / (Distance * Distance);
        vec3 Radiance = lightColor[i] * Attenuation;

        float NDF = distributionGGX(normalW, HalfVector, roughness);
        float G = geometrySmith(normalW, viewDirW, LightDirection, roughness);
        vec3 F = fresnelSchlick(clamp(dot(HalfVector, viewDirW), 0.0f, 1.0f), F0);

        vec3 Nominator = NDF * G * F;
        float Denominator = 4 * max(dot(normalW, viewDirW), 0.0f) * max(dot(normalW, LightDirection), 0.0f);
        vec3 Specular = Nominator / max(Denominator, 0.001f);

        vec3 KS = F;
        vec3 KD = (1.0 - metallic) * (vec3(1.0f) - KS);
        float Ndotl = max(dot(normalW, LightDirection), 0.0f);

        Lo += (KD * albedo / PI + Specular) * Radiance * Ndotl;
    }

    return Lo;
}


vec3 shadingWithUnrealModelForImageBasedLight(vec3 albedo, float metallic, float roughness, vec3 normalW, vec3 viewDirW)
{
    const vec3 ReflectDirW = reflect(-viewDirW, normalW);
    vec3 F0 = mix(vec3(0.04f), albedo, metallic);

    const float MAX_REFLECTION_LOD = 11.0f;
    const float CosTheta = max(dot(normalW, viewDirW), 0.0f);
    vec3 F = fresnelSchlickRougness(CosTheta, F0, roughness);

    vec3 Irradiance = textureLod(uIrradianceMap, normalW, 0).xyz;
    vec3 Diffuse = Irradiance * albedo;

    vec3 PrefilteredColor = textureLod(uPrefilteredMap, ReflectDirW, roughness * MAX_REFLECTION_LOD).xyz;
    vec2 CoordForBrdf = vec2(CosTheta, roughness);
    vec2 BrdfValue = texture(uBrdfIntegrationMap, CoordForBrdf).xy;
    vec3 Specular = PrefilteredColor * (F * BrdfValue.x + BrdfValue.y);

    const vec3 KS = F;
    const vec3 KD = (1.0f - metallic) * (vec3(1.0f) - KS);
    vec3 LoForIBL = KD * Diffuse + Specular;

    return LoForIBL;
}


vec3 calcLightInternal(SBaseLight baseLight, vec3 lightDirection,  vec3 positionW, vec3 normal)
{
	vec3 AmbientColor = baseLight.Color.xyz * baseLight.AmbientIntensity;
	vec3 DiffuseColor = vec3(0.0f);
	vec3 SpecularColor = vec3(0.0f);

	vec3 LightDirection = normalize(-lightDirection);
	float DiffuseFactor = dot(normalize(normal), LightDirection);
	if (DiffuseFactor > 0.0f)
	{
		DiffuseColor = baseLight.Color.xyz * baseLight.DiffuseIntensity * DiffuseFactor;

		vec3 Vertex2Eye = normalize(uCameraPos - positionW);
		vec3 ReflectLight = normalize(reflect(-LightDirection, normalize(normal)));

		float Specularvactor = dot(Vertex2Eye, ReflectLight);
		if (Specularvactor > 0.0f)
		{
			SpecularColor = pow(Specularvactor, uShininess) * baseLight.Color.xyz * uSpecularIntensity;
		}
	}

	return AmbientColor + DiffuseColor + SpecularColor;
}